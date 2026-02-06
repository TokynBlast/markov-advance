use anyhow::Result;
use crossbeam_channel::unbounded;
use flate2::read::GzDecoder;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::mpsc::{self, Sender, Receiver};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use tar::Archive;
use regex::Regex;
use zip::ZipArchive;

/// Transition weight can be 8-bit (fast), 32-bit (precise), 64-bit (double), or TRUE 128-bit (u128)
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
enum TransitionWeight {
    Quantized(u8),           // 8-bit: 0-255 → 0.0-1.0
    HighPrecision(f32),      // 32-bit: full float precision
    ExtremePrecision(f64),   // 64-bit double precision
    UltraPrecision(u128),    // TRUE 128-bit: 0 to 340_282_366_920_938_463_463_374_607_431_768_211_455
}

impl TransitionWeight {
    fn to_f32(&self) -> f32 {
        match self {
            TransitionWeight::Quantized(q) => (*q as f32) / 255.0,
            TransitionWeight::HighPrecision(f) => *f,
            TransitionWeight::ExtremePrecision(d) => *d as f32,
            TransitionWeight::UltraPrecision(u) => (*u as f64 / u128::MAX as f64) as f32,
        }
    }

    fn from_f32(value: f32, precision_mode: u16) -> Self {
        match precision_mode {
            8 => TransitionWeight::Quantized(((value.max(0.0).min(1.0)) * 255.0) as u8),
            64 => TransitionWeight::HighPrecision(value.max(0.0).min(1.0)),
            128 => TransitionWeight::ExtremePrecision(value.max(0.0).min(1.0) as f64),
            256 => {
                // TRUE 128-bit mode: scale to full u128 range
                let scaled = (value.max(0.0).min(1.0) as f64 * u128::MAX as f64) as u128;
                TransitionWeight::UltraPrecision(scaled)
            }
            _ => TransitionWeight::Quantized(((value.max(0.0).min(1.0)) * 255.0) as u8),
        }
    }
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "is", "are", "was", "were", "be", "been", "being", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "as", "not", "no",
    "yes", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might",
];

const CONTEXT_SIZE: usize = 150;
const SUBJECT_WINDOW: usize = 150;
const CONVERSATION_HISTORY_CAP: usize = 500;
const PROGRESS_EVERY_WORDS: usize = 50_000;
const LARGE_FILE_BYTES: usize = 500_000;
const CHUNK_LINES: usize = 1000;
const PARALLEL_MERGE_MIN: usize = 10_000;
const GPU_NORMALIZE_MIN_WEIGHTS: usize = 200_000;
const GPU_WORKGROUP_SIZE: u32 = 256;
const STYLE_TOKEN_CHANCE: f32 = 0.08;
const CONTEXT_PAIR_LIMIT: usize = 50;
const CONTEXT_PAIR_SEARCH_LIMIT: usize = 100;
const CONTEXT_PAIR_FORWARD_SEARCH_LIMIT: usize = 150;
const RELEVANCE_SCAN_WINDOW: usize = 15;
const RELEVANCE_WINDOW_SIZE: usize = 50;
const DEFAULT_GENERATE_COUNT: usize = 50;
const DEFAULT_TALK_COUNT: usize = 40;

const GPU_NORMALIZE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> weights: array<f32>;

@group(0) @binding(1)
var<storage, read> word_index: array<u32>;

@group(0) @binding(2)
var<storage, read> totals: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&weights)) {
        return;
    }
    let idx = word_index[i];
    let total = totals[idx];
    if (total > 0.0) {
        weights[i] = weights[i] / total;
    }
}
"#;

struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBackend {
    fn init(preference: &str) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let pref = preference.to_ascii_lowercase();

        let adapter = if pref == "auto" {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        } else {
            let vendor = match pref.as_str() {
                "nvidia" => Some(0x10de),
                "amd" => Some(0x1002),
                "intel" => Some(0x8086),
                _ => None,
            };

            let mut picked = None;
            for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                let info = adapter.get_info();
                let name = info.name.to_ascii_lowercase();
                let vendor_match = vendor.map(|v| info.vendor == v).unwrap_or(false);
                if vendor_match || name.contains(&pref) {
                    picked = Some(adapter);
                    break;
                }
            }

            picked.or_else(|| {
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
            })
        }?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normalize-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GPU_NORMALIZE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normalize-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normalize-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("normalize-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    fn normalize_weights(&self, weights: &mut [f32], word_index: &[u32], totals: &[f32]) -> bool {
        if weights.is_empty() {
            return false;
        }

        let weights_bytes = bytemuck::cast_slice(weights);
        let index_bytes = bytemuck::cast_slice(word_index);
        let totals_bytes = bytemuck::cast_slice(totals);

        let weights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights-buffer"),
            size: weights_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&weights_buffer, 0, weights_bytes);

        let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index-buffer"),
            size: index_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&index_buffer, 0, index_bytes);

        let totals_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("totals-buffer"),
            size: totals_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&totals_buffer, 0, totals_bytes);

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights-readback"),
            size: weights_bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normalize-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: totals_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("normalize-encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normalize-pass"),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (weights.len() as u32 + GPU_WORKGROUP_SIZE - 1) / GPU_WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&weights_buffer, 0, &readback, 0, weights_bytes.len() as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = readback.slice(..);
        if pollster::block_on(slice.map_async(wgpu::MapMode::Read)).is_err() {
            return false;
        }

        let data = slice.get_mapped_range();
        let updated = bytemuck::cast_slice(&data);
        if updated.len() != weights.len() {
            return false;
        }
        weights.copy_from_slice(updated);
        drop(data);
        readback.unmap();
        true
    }
}

fn is_punct(token: &str) -> bool {
    matches!(token, "." | "," | "!" | "?" | ";" | ":")
}

fn is_quote_token(token: &str) -> bool {
    token == "\""
}

fn is_word_token(token: &str) -> bool {
    token.chars().any(|c| c.is_ascii_alphanumeric())
}

fn is_style_token(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    if lower == ":3" || lower == ":)" || lower == ":d" || lower == ";)" {
        return true;
    }
    token.chars().any(|c| !c.is_ascii_alphanumeric()) && token.len() <= 4
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphanumeric() || c == '\'' {
            current.push(c);
        } else if c == '"' {
            if !current.is_empty() {
                tokens.push(current.to_ascii_lowercase());
                current.clear();
            }
            tokens.push("\"".to_string());
        } else if matches!(c, '.' | ',' | '!' | '?' | ';' | ':') {
            if !current.is_empty() {
                tokens.push(current.to_ascii_lowercase());
                current.clear();
            }

            // Detect simple emoticons like :3, :), :D
            if c == ':' && i + 1 < chars.len() {
                let next = chars[i + 1];
                if matches!(next, '3' | ')' | 'D' | 'd') {
                    let emoticon = format!(":{}", next);
                    tokens.push(emoticon.to_ascii_lowercase());
                    i += 1;
                } else {
                    tokens.push(c.to_string());
                }
            } else {
                tokens.push(c.to_string());
            }
        } else if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.to_ascii_lowercase());
                current.clear();
            }
        } else {
            current.push(c);
        }
        i += 1;
    }

    if !current.is_empty() {
        tokens.push(current.to_ascii_lowercase());
    }

    tokens
}

/// Tokenize preserving original capitalization (for grammar awareness)
fn tokenize_with_case(text: &str) -> Vec<(String, String)> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphanumeric() || c == '\'' {
            current.push(c);
        } else if c == '"' {
            if !current.is_empty() {
                let original = current.clone();
                let lowercase = current.to_ascii_lowercase();
                tokens.push((lowercase, original));
                current.clear();
            }
            tokens.push(("\"".to_string(), "\"".to_string()));
        } else if matches!(c, '.' | ',' | '!' | '?' | ';' | ':') {
            if !current.is_empty() {
                let original = current.clone();
                let lowercase = current.to_ascii_lowercase();
                tokens.push((lowercase, original));
                current.clear();
            }

            // Detect simple emoticons like :3, :), :D
            if c == ':' && i + 1 < chars.len() {
                let next = chars[i + 1];
                if matches!(next, '3' | ')' | 'D' | 'd') {
                    let emoticon = format!(":{}", next);
                    tokens.push((emoticon.to_ascii_lowercase(), emoticon));
                    i += 1;
                } else {
                    let p = c.to_string();
                    tokens.push((p.clone(), p));
                }
            } else {
                let p = c.to_string();
                tokens.push((p.clone(), p));
            }
        } else if c.is_whitespace() {
            if !current.is_empty() {
                let original = current.clone();
                let lowercase = current.to_ascii_lowercase();
                tokens.push((lowercase, original));
                current.clear();
            }
        } else {
            current.push(c);
        }
        i += 1;
    }

    if !current.is_empty() {
        let original = current.clone();
        let lowercase = current.to_ascii_lowercase();
        tokens.push((lowercase, original));
    }

    tokens
}

fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    let mut prev_was_punct = false;
    let mut in_quote = false;

    for token in tokens {
        if is_quote_token(token) {
            if in_quote {
                out.push_str(token);
                in_quote = false;
            } else {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
                out.push_str(token);
                in_quote = true;
            }
            prev_was_punct = false;
            continue;
        }

        if is_punct(token) {
            if prev_was_punct {
                let last = out.chars().last().unwrap_or(' ');
                let current = token.chars().next().unwrap_or(' ');
                if last == current {
                    out.push_str(token);
                }
            } else {
                out.push_str(token);
            }
            prev_was_punct = true;
            continue;
        }

        let needs_space = !out.is_empty()
            && !out.ends_with(' ')
            && !(in_quote && out.ends_with('"'));
        if needs_space {
            out.push(' ');
        }
        out.push_str(token);
        prev_was_punct = false;
    }
    out
}

fn is_supported_training_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    name.ends_with(".txt")
        || name.ends_with(".gz")
        || name.ends_with(".tgz")
        || name.ends_with(".tar.gz")
        || name.ends_with(".tar")
        || name.ends_with(".zip")
}

fn read_text_from_reader<R: Read>(mut reader: R) -> Vec<String> {
    let mut buf = Vec::new();
    if reader.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }

    match String::from_utf8(buf) {
        Ok(text) if !text.trim().is_empty() => vec![text],
        _ => Vec::new(),
    }
}

fn collect_tar_texts<R: Read>(archive: &mut Archive<R>) -> Vec<String> {
    let mut texts = Vec::new();
    let entries = match archive.entries() {
        Ok(entries) => entries,
        Err(_) => return texts,
    };

    for entry in entries {
        let mut entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };

        let path = match entry.path() {
            Ok(path) => path.to_path_buf(),
            Err(_) => continue,
        };

        if path.extension().and_then(|s| s.to_str()).map_or(true, |ext| ext != "txt") {
            continue;
        }

        let mut buf = Vec::new();
        if entry.read_to_end(&mut buf).is_ok() {
            if let Ok(text) = String::from_utf8(buf) {
                if !text.trim().is_empty() {
                    texts.push(text);
                }
            }
        }
    }

    texts
}

fn collect_zip_texts(archive: &mut ZipArchive<File>) -> Vec<String> {
    let mut texts = Vec::new();
    for i in 0..archive.len() {
        let mut file = match archive.by_index(i) {
            Ok(file) => file,
            Err(_) => continue,
        };

        if file.is_dir() {
            continue;
        }

        let name = file.name().to_ascii_lowercase();
        if !name.ends_with(".txt") {
            continue;
        }

        let mut buf = Vec::new();
        if file.read_to_end(&mut buf).is_ok() {
            if let Ok(text) = String::from_utf8(buf) {
                if !text.trim().is_empty() {
                    texts.push(text);
                }
            }
        }
    }
    texts
}

fn read_training_texts(path: &Path) -> Result<Vec<String>> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if name.ends_with(".txt") {
        return Ok(vec![fs::read_to_string(path)?]);
    }

    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        return Ok(collect_tar_texts(&mut archive));
    }

    if name.ends_with(".tar") {
        let file = File::open(path)?;
        let mut archive = Archive::new(file);
        return Ok(collect_tar_texts(&mut archive));
    }

    if name.ends_with(".zip") {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        return Ok(collect_zip_texts(&mut archive));
    }

    if name.ends_with(".gz") {
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        return Ok(read_text_from_reader(decoder));
    }

    Ok(Vec::new())
}

fn load_nono_words(path: &str) -> FxHashSet<String> {
    let content = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return FxHashSet::default(),
    };

    let mut words = FxHashSet::default();
    for line in content.lines() {
        let raw = line.split('#').next().unwrap_or("");
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        for token in trimmed.split_whitespace() {
            let lower = token.to_ascii_lowercase();
            if !lower.is_empty() {
                words.insert(lower);
            }
        }
    }
    words
}

fn filter_tokens_with_case(
    tokens: Vec<(String, String)>,
    banned: &FxHashSet<String>,
) -> Vec<(String, String)> {
    if banned.is_empty() {
        return tokens;
    }

    tokens
        .into_iter()
        .filter(|(lower, _)| {
            if !is_word_token(lower) {
                return true;
            }
            !banned.contains(lower)
        })
        .collect()
}

fn count_tokens_filtered(text: &str, banned: &FxHashSet<String>) -> usize {
    let tokens = tokenize_with_case(text);
    let filtered = filter_tokens_with_case(tokens, banned);
    filtered.len()
}

/// GRAMMAR HELPER: Capitalize the first letter of a word
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Source Purifier - Remove data-poisoning elements from training text
/// Cleans: URLs, markdown, forewords, navigation, Project Gutenberg boilerplate, etc.
fn purify_text(text: &str) -> String {
    let mut cleaned = text.to_string();

    // Remove URLs (http, https, www)
    let url_pattern = Regex::new(r"https?://[^\s]+|www\.[^\s]+").unwrap();
    cleaned = url_pattern.replace_all(&cleaned, "").to_string();

    // Remove markdown links: [text](url) -> text
    let md_link_pattern = Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap();
    cleaned = md_link_pattern.replace_all(&cleaned, "$1").to_string();

    // Remove markdown images: ![alt](url)
    let md_image_pattern = Regex::new(r"!\[([^\]]*)\]\([^\)]+\)").unwrap();
    cleaned = md_image_pattern.replace_all(&cleaned, "").to_string();

    // Remove markdown headers: ### Header -> Header
    let md_header_pattern = Regex::new(r"^#{1,6}\s+(.+)$").unwrap();
    cleaned = md_header_pattern.replace_all(&cleaned, "$1").to_string();

    // Remove markdown code blocks: ```code```
    let md_code_pattern = Regex::new(r"```[\s\S]*?```").unwrap();
    cleaned = md_code_pattern.replace_all(&cleaned, "").to_string();

    // Remove inline code: `code` -> code
    let md_inline_code = Regex::new(r"`([^`]+)`").unwrap();
    cleaned = md_inline_code.replace_all(&cleaned, "$1").to_string();

    // Remove email addresses
    let email_pattern = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();
    cleaned = email_pattern.replace_all(&cleaned, "").to_string();

    // Remove Project Gutenberg boilerplate (common poison)
    let gutenberg_patterns = [
        r"(?i)project gutenberg.*?etext",
        r"(?i)start of (?:the )?project gutenberg.*",
        r"(?i)end of (?:the )?project gutenberg.*",
        r"(?i)produced by.*",
        r"(?i)this etext.*prepared by.*",
        r"(?i)gutenberg.*license",
        r"(?i)table of contents",
        r"(?i)chapter [ivxlcdm0-9]+\.?\s*$",
    ];

    for pattern_str in &gutenberg_patterns {
        if let Ok(pattern) = Regex::new(pattern_str) {
            cleaned = pattern.replace_all(&cleaned, "").to_string();
        }
    }

    // Remove common navigation/UI elements
    let nav_patterns = [
        r"(?i)click here",
        r"(?i)read more",
        r"(?i)continue reading",
        r"(?i)next page",
        r"(?i)previous page",
        r"(?i)back to top",
        r"\[edit\]",
        r"\[citation needed\]",
    ];

    for pattern_str in &nav_patterns {
        if let Ok(pattern) = Regex::new(pattern_str) {
            cleaned = pattern.replace_all(&cleaned, "").to_string();
        }
    }

    // Remove excessive whitespace
    let whitespace_pattern = Regex::new(r"\s+").unwrap();
    cleaned = whitespace_pattern.replace_all(&cleaned, " ").to_string();

    // Remove lines that are just numbers (page numbers, etc.)
    let lines: Vec<&str> = cleaned.lines()
        .filter(|line| {
            let trimmed = line.trim();
            // Skip if line is just a number or roman numeral
            !trimmed.chars().all(|c| c.is_numeric() || c.is_whitespace())
        })
        .collect();

    cleaned = lines.join("\n");

    cleaned.trim().to_string()
}

fn build_subject_weights(context: &VecDeque<String>) -> FxHashMap<String, f32> {
    let mut scores: FxHashMap<String, f32> = FxHashMap::default();
    for token in context.iter().rev().take(SUBJECT_WINDOW) {
        if !is_word_token(token) {
            continue;
        }
        if STOPWORDS.contains(&token.as_str()) {
            continue;
        }
        let entry = scores.entry(token.clone()).or_insert(0.0);
        *entry += 1.0;
    }

    let max = scores.values().cloned().fold(0.0, f32::max).max(1.0);
    for value in scores.values_mut() {
        *value /= max;
    }
    scores
}

/// Build subject weights with higher priority for user messages
fn build_subject_weights_with_user_boost(context: &VecDeque<String>, user_token_count: usize) -> FxHashMap<String, f32> {
    let mut scores: FxHashMap<String, f32> = FxHashMap::default();

    // Process most recent tokens (mostly user's last message + bot response)
    // Give recent tokens (especially user input) much higher weight
    let mut idx = 0;
    for token in context.iter().rev().take(SUBJECT_WINDOW) {
        if !is_word_token(token) {
            idx += 1;
            continue;
        }
        if STOPWORDS.contains(&token.as_str()) {
            idx += 1;
            continue;
        }

        // If this token is within the user message (recent), boost it heavily
        let boost = if idx < user_token_count { 2.5 } else { 1.0 };

        let entry = scores.entry(token.clone()).or_insert(0.0);
        *entry += boost;
        idx += 1;
    }

    let max = scores.values().cloned().fold(0.0, f32::max).max(1.0);
    for value in scores.values_mut() {
        *value /= max;
    }
    scores
}

fn update_context(context: &mut VecDeque<String>, tokens: &[String], max_size: usize) {
    for token in tokens {
        context.push_back(token.clone());
        if context.len() > max_size {
            context.pop_front();
        }
    }
}

fn update_style_counts(style_counts: &mut FxHashMap<String, u32>, tokens: &[String]) {
    for token in tokens {
        if is_style_token(token) {
            *style_counts.entry(token.clone()).or_insert(0) += 1;
        }
    }
}

fn pick_style_token(style_counts: &FxHashMap<String, u32>, rng: &mut impl Rng) -> Option<String> {
    if style_counts.is_empty() {
        return None;
    }

    if rng.r#gen::<f32>() > STYLE_TOKEN_CHANCE {
        return None;
    }

    let mut pool: Vec<(&String, u32)> = style_counts.iter().map(|(k, v)| (k, *v)).collect();
    pool.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let top = pool.iter().take(5).collect::<Vec<_>>();
    let choice = top.choose(rng)?;
    Some((*choice.0).clone())
}

/// Semantic similarity between two words (0.0 to 1.0)
fn semantic_similarity(word1: &str, word2: &str) -> f32 {
    if word1 == word2 {
        return 1.0;
    }

    let word1 = word1.to_ascii_lowercase();
    let word2 = word2.to_ascii_lowercase();

    // Prefix overlap: "running" and "runs" should be similar
    let min_len = word1.len().min(word2.len());
    if min_len >= 3 {
        let mut prefix_overlap = 0;
        for (c1, c2) in word1.chars().zip(word2.chars()) {
            if c1 == c2 {
                prefix_overlap += 1;
            } else {
                break;
            }
        }
        if prefix_overlap >= 3 {
            return (prefix_overlap as f32) / (word1.len().max(word2.len()) as f32);
        }
    }

    // Suffix overlap
    let suffix_overlap = word1.chars().rev().zip(word2.chars().rev())
        .take_while(|(c1, c2)| c1 == c2)
        .count();
    if suffix_overlap >= 2 {
        return (suffix_overlap as f32) / (word1.len().max(word2.len()) as f32);
    }

    0.0
}

/// Find semantically similar words in the context
fn find_semantic_cluster(word: &str, context: &[String]) -> Vec<String> {
    let mut cluster = vec![word.to_string()];
    for ctx_word in context {
        if semantic_similarity(word, ctx_word) >= 0.4 && !cluster.contains(ctx_word) {
            cluster.push(ctx_word.clone());
        }
    }
    cluster
}

/// Background relevance scanner - analyzes context in parallel to find most relevant topics
/// Scans RELEVANCE_SCAN_WINDOW-token windows within the past RELEVANCE_WINDOW_SIZE words
struct RelevanceScanner {
    context_rx: Receiver<Vec<String>>,
    query_tx: Sender<Option<String>>,
    query_rx: Receiver<()>,
}

impl RelevanceScanner {
    /// Spawn a background scanner thread
    fn spawn() -> (Sender<Vec<String>>, Sender<()>, Receiver<Option<String>>) {
        let (context_tx, context_rx) = mpsc::channel();
        let (query_tx_internal, query_rx_internal) = mpsc::channel();
        let (query_req_tx, query_req_rx) = mpsc::channel();

        thread::spawn(move || {
            let scanner = RelevanceScanner {
                context_rx,
                query_tx: query_tx_internal,
                query_rx: query_req_rx,
            };
            scanner.run();
        });

        (context_tx, query_req_tx, query_rx_internal)
    }

    fn run(self) {
        let mut current_context: Vec<String> = Vec::new();

        loop {
            // Update context if available
            if let Ok(new_context) = self.context_rx.try_recv() {
                current_context = new_context;
            }

            // Respond to relevance queries
            if let Ok(()) = self.query_rx.try_recv() {
                let relevant_topic = self.find_most_relevant(&current_context);
                let _ = self.query_tx.send(relevant_topic);
            }

            // Small sleep to avoid busy-waiting
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn find_most_relevant(&self, context: &[String]) -> Option<String> {
        if context.is_empty() {
            return None;
        }

        // Take last RELEVANCE_WINDOW_SIZE words
        let window_size = RELEVANCE_WINDOW_SIZE.min(context.len());
        let recent = &context[context.len().saturating_sub(window_size)..];

        // Scan RELEVANCE_SCAN_WINDOW-token windows at random positions (but in order)
        let chunk_size = RELEVANCE_SCAN_WINDOW;
        let mut topic_scores: FxHashMap<String, f32> = FxHashMap::default();

        for start in (0..recent.len()).step_by(5) {
            if start + chunk_size > recent.len() {
                break;
            }

            let chunk = &recent[start..start + chunk_size.min(recent.len() - start)];

            // Calculate word importance in this chunk
            for word in chunk {
                if STOPWORDS.contains(&word.as_str()) {
                    continue;
                }

                // Score based on local frequency and position
                let freq = chunk.iter().filter(|w| *w == word).count() as f32;
                let recency = 1.0 - (start as f32 / recent.len() as f32);
                let score = freq * (1.0 + recency * 0.5);

                *topic_scores.entry(word.clone()).or_insert(0.0) += score;
            }
        }

        // Find highest-scoring topic
        topic_scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(topic, _)| topic)
    }
}

/// Advanced Nudged Markov Bot with bidirectional context awareness + GRAMMAR INTELLIGENCE
#[derive(Serialize, Deserialize, Clone)]
struct NudgeBot {
    /// Bigram transitions: word -> [(next_word, weight)]
    transitions: FxHashMap<String, Vec<(String, TransitionWeight)>>,
    /// Skip-gram context: word -> [(related_word, co_occurrence_count)]
    context_pairs: FxHashMap<String, Vec<(String, u32)>>,
    /// Word frequency
    frequency: FxHashMap<String, u32>,
    /// Context window size
    context_size: usize,
    /// Precision mode: 8 (quantized), 64 (f32), 128 (f64), 256 (u128)
    precision_mode: u16,

    // === GRAMMAR INTELLIGENCE ===
    /// Track capitalization: lowercase_word -> (capitalized_count, lowercase_count)
    capitalization: FxHashMap<String, (u32, u32)>,
    /// Words that commonly start sentences
    sentence_starters: FxHashMap<String, u32>,
    /// What words follow punctuation (for natural breaks)
    after_punctuation: FxHashMap<String, u32>,
}

/// Frozen model with index-based storage (for perfect-hash lookup)
#[derive(Serialize, Deserialize, Clone)]
struct FrozenModel {
    vocab: Vec<String>,
    transitions: Vec<Vec<(u32, TransitionWeight)>>,
    context_pairs: Vec<Vec<(u32, u32)>>,
    frequency: Vec<u32>,
    capitalization: Vec<(u32, u32)>,
    sentence_starters: Vec<u32>,
    after_punctuation: Vec<u32>,
    context_size: usize,
    precision_mode: u16,
}

impl FrozenModel {
    fn from_bot(bot: &NudgeBot) -> Self {
        let mut vocab: Vec<String> = bot.frequency.keys().cloned().collect();
        vocab.sort();

        let mut index: FxHashMap<String, u32> = FxHashMap::default();
        for (i, word) in vocab.iter().enumerate() {
            index.insert(word.clone(), i as u32);
        }

        let mut transitions: Vec<Vec<(u32, TransitionWeight)>> = vec![Vec::new(); vocab.len()];
        for (word, list) in &bot.transitions {
            if let Some(&i) = index.get(word) {
                for (next, weight) in list {
                    if let Some(&j) = index.get(next) {
                        transitions[i as usize].push((j, *weight));
                    }
                }
            }
        }

        let mut context_pairs: Vec<Vec<(u32, u32)>> = vec![Vec::new(); vocab.len()];
        for (word, list) in &bot.context_pairs {
            if let Some(&i) = index.get(word) {
                for (ctx, count) in list {
                    if let Some(&j) = index.get(ctx) {
                        context_pairs[i as usize].push((j, *count));
                    }
                }
            }
        }

        let mut frequency: Vec<u32> = vec![0; vocab.len()];
        let mut capitalization: Vec<(u32, u32)> = vec![(0, 0); vocab.len()];
        let mut sentence_starters: Vec<u32> = vec![0; vocab.len()];
        let mut after_punctuation: Vec<u32> = vec![0; vocab.len()];

        for (i, word) in vocab.iter().enumerate() {
            if let Some(count) = bot.frequency.get(word) {
                frequency[i] = *count;
            }
            if let Some(counts) = bot.capitalization.get(word) {
                capitalization[i] = *counts;
            }
            if let Some(count) = bot.sentence_starters.get(word) {
                sentence_starters[i] = *count;
            }
            if let Some(count) = bot.after_punctuation.get(word) {
                after_punctuation[i] = *count;
            }
        }

        Self {
            vocab,
            transitions,
            context_pairs,
            frequency,
            capitalization,
            sentence_starters,
            after_punctuation,
            context_size: bot.context_size,
            precision_mode: bot.precision_mode,
        }
    }

    fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::serialize(self)?;
        let compressed = zstd::encode_all(&serialized[..], 10)?;
        fs::write(path, compressed)?;
        println!("[✓] Frozen model saved: {}", path);
        Ok(())
    }

    fn load(path: &str) -> Result<Self> {
        let compressed = fs::read(path)?;
        let serialized = zstd::decode_all(&compressed[..])?;
        let model = bincode::deserialize(&serialized)?;
        println!("[✓] Frozen model loaded");
        Ok(model)
    }

    fn index_of(&self, word: &str) -> Option<usize> {
        self.vocab.binary_search_by(|w| w.as_str().cmp(word)).ok()
    }

    fn apply_grammar_index(&self, word: &str, idx: usize, should_capitalize: bool, words_since_punct: usize) -> String {
        if is_punct(word) {
            return word.to_string();
        }

        let (cap_count, lower_count) = self.capitalization[idx];
        let total = cap_count + lower_count;
        if total == 0 {
            return word.to_string();
        }

        let cap_ratio = cap_count as f32 / total as f32;
        let usually_capitalized = cap_ratio > 0.7;

        if usually_capitalized {
            return capitalize_first(word);
        }

        if should_capitalize {
            return capitalize_first(word);
        }

        if self.sentence_starters[idx] > 0 && words_since_punct == 0 {
            return capitalize_first(word);
        }

        word.to_string()
    }

    fn build_subject_weights_vec(&self, context: &VecDeque<String>, user_token_count: usize) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.vocab.len()];
        let mut idx = 0;
        for token in context.iter().rev().take(SUBJECT_WINDOW) {
            if !is_word_token(token) {
                idx += 1;
                continue;
            }
            if STOPWORDS.contains(&token.as_str()) {
                idx += 1;
                continue;
            }

            if let Some(word_idx) = self.index_of(token) {
                let boost = if idx < user_token_count { 2.5 } else { 1.0 };
                scores[word_idx] += boost;
            }
            idx += 1;
        }

        let max = scores.iter().cloned().fold(0.0, f32::max).max(1.0);
        for value in scores.iter_mut() {
            *value /= max;
        }
        scores
    }

    fn apply_context_nudges_index(
        &self,
        current_idx: usize,
        context_before: &VecDeque<String>,
        context_after: &VecDeque<String>,
        subject_weights: &[f32],
    ) -> Vec<(u32, f32)> {
        let transitions = &self.transitions[current_idx];
        if transitions.is_empty() {
            return vec![];
        }

        let mut nudged: Vec<(u32, f32)> = transitions
            .iter()
            .map(|(idx, w)| (*idx, w.to_f32()))
            .collect();

        let all_context: Vec<String> = context_before.iter().chain(context_after.iter()).cloned().collect();
        for (next_idx, weight) in nudged.iter_mut() {
            let next_word = &self.vocab[*next_idx as usize];

            let recent_window = all_context.iter().rev().take(20);
            let older_context = all_context.iter().rev().skip(20);

            let recent_exact = recent_window.clone().filter(|w| *w == next_word).count() as f32;
            let older_exact = older_context.clone().filter(|w| *w == next_word).count() as f32;

            let exact_count = recent_exact * 5.0 + older_exact;

            let mut recent_semantic = 0.0;
            let mut older_semantic = 0.0;

            for ctx_word in recent_window {
                let similarity = semantic_similarity(next_word, ctx_word);
                if similarity >= 0.3 {
                    recent_semantic += similarity;
                }
            }

            for ctx_word in older_context {
                let similarity = semantic_similarity(next_word, ctx_word);
                if similarity >= 0.3 {
                    older_semantic += similarity;
                }
            }

            let semantic_count = recent_semantic * 3.0 + older_semantic;
            let total_penalty = exact_count * 0.9 + semantic_count * 0.6;
            if total_penalty > 0.0 {
                *weight /= (1.0 + total_penalty).powf(1.5);
            }

            // Extra anti-repeat: penalize recent exact matches harder
            if let Some(last_token) = context_before.back() {
                if last_token == next_word {
                    *weight *= 0.05;
                }
            }
            if context_before.iter().rev().take(8).any(|w| w == next_word) {
                *weight *= 0.2;
            }
        }

        for (distance, ctx_word) in context_before.iter().rev().take(10).enumerate() {
            let distance_factor = 1.0 / (1.0 + distance as f32 * 0.3);
            if let Some(ctx_idx) = self.index_of(ctx_word) {
                let ctx_transitions = &self.transitions[ctx_idx];
                let ctx_pairs = &self.context_pairs[ctx_idx];

                for (next_idx, base_weight) in nudged.iter_mut() {
                    if let Some((_, ctx_weight)) = ctx_transitions.iter().find(|(w, _)| w == next_idx) {
                        let support = ctx_weight.to_f32();
                        let nudge = support * distance_factor * 0.11;
                        *base_weight = (*base_weight + nudge).min(1.0);
                    }

                    if ctx_pairs.len() < CONTEXT_PAIR_SEARCH_LIMIT {
                        if let Some((_, count)) = ctx_pairs.iter().find(|(w, _)| w == next_idx) {
                            let pair_strength = (*count as f32 / 100.0).min(1.0);
                            let nudge = pair_strength * distance_factor * 0.08;
                            *base_weight = (*base_weight + nudge).min(1.0);
                        }
                    }
                }
            }
        }

        for (distance, ctx_word) in context_after.iter().take(10).enumerate() {
            let distance_factor = 1.0 / (1.0 + distance as f32 * 0.5);
            if let Some(ctx_idx) = self.index_of(ctx_word) {
                let ctx_pairs = &self.context_pairs[ctx_idx];

                for (next_idx, base_weight) in nudged.iter_mut() {
                    if ctx_pairs.len() < CONTEXT_PAIR_SEARCH_LIMIT {
                        if let Some((_, count)) = ctx_pairs.iter().find(|(w, _)| w == next_idx) {
                            let pair_strength = (*count as f32 / 100.0).min(1.0);
                            let nudge = pair_strength * distance_factor * 0.05;
                            *base_weight = (*base_weight + nudge).min(1.0);
                        }
                    }
                }
            }
        }

        for (next_idx, base_weight) in nudged.iter_mut() {
            let score = subject_weights.get(*next_idx as usize).copied().unwrap_or(0.0);
            let nudge = (score * 0.30).min(0.30);
            *base_weight = (*base_weight + nudge).min(1.0);
        }

        let total: f32 = nudged.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            for (_, weight) in nudged.iter_mut() {
                *weight /= total;
            }
        }

        nudged
    }

    fn generate(&self, count: usize, seed_word: Option<&str>) -> String {
        self.generate_with_pass(count, seed_word, false)
    }

    fn generate_with_pass(&self, count: usize, pass_text: Option<&str>, stream: bool) -> String {
        let mut rng = rand::thread_rng();
        let mut output: Vec<String> = Vec::new();
        let mut local_context: VecDeque<String> = VecDeque::new();
        let context_after: VecDeque<String> = VecDeque::with_capacity(self.context_size);
        let mut stream_first = true;
        let mut stream_last: Option<String> = None;
        let mut stream_in_quote = false;

        let mut current_idx = if let Some(pass) = pass_text {
            let pass_tokens = tokenize_with_case(pass);
            for (lower, original) in &pass_tokens {
                output.push(original.clone());
                if stream {
                    stream_token(original, &mut stream_first, &mut stream_last, &mut stream_in_quote);
                }
                update_context(&mut local_context, &[lower.clone()], self.context_size);
            }
            if let Some((last_lower, _)) = pass_tokens.last() {
                self.index_of(last_lower).unwrap_or(0)
            } else {
                0
            }
        } else if !self.vocab.is_empty() {
            rng.gen_range(0..self.vocab.len())
        } else {
            return String::new();
        };

        if pass_text.is_none() {
            let start_word = self.vocab[current_idx].clone();
            let formatted_start = self.apply_grammar_index(&start_word, current_idx, true, 0);
            output.push(formatted_start.clone());
            if stream {
                stream_token(&formatted_start, &mut stream_first, &mut stream_last, &mut stream_in_quote);
            }
            update_context(&mut local_context, &[start_word.clone()], self.context_size);
        }

        let mut should_capitalize = match local_context.back() {
            Some(tok) if matches!(tok.as_str(), "." | "!" | "?") => true,
            _ => false,
        };
        let mut words_since_punct = 0;
        if !local_context.is_empty() {
            for tok in local_context.iter().rev() {
                if matches!(tok.as_str(), "." | "!" | "?") {
                    break;
                }
                if !is_punct(tok) {
                    words_since_punct += 1;
                }
            }
        }

        let mut subject_weights = self.build_subject_weights_vec(&local_context, 0);
        let mut steps_since_subject = 0usize;

        while output.len() < count {
            if steps_since_subject >= 5 {
                subject_weights = self.build_subject_weights_vec(&local_context, 0);
                steps_since_subject = 0;
            }
            let candidates = self.apply_context_nudges_index(
                current_idx,
                &local_context,
                &context_after,
                &subject_weights,
            );

            if candidates.is_empty() {
                break;
            }

            let weights: Vec<f32> = candidates.iter().map(|(_, w)| *w).collect();
            if let Some(choice) = weighted_choice(&weights, &mut rng) {
                let next_idx = candidates[choice].0 as usize;
                let next_word = self.vocab[next_idx].clone();
                let formatted = self.apply_grammar_index(&next_word, next_idx, should_capitalize, words_since_punct);

                output.push(formatted.clone());
                if stream {
                    stream_token(&formatted, &mut stream_first, &mut stream_last, &mut stream_in_quote);
                }
                update_context(&mut local_context, &[next_word.clone()], self.context_size);
                current_idx = next_idx;

                if matches!(next_word.as_str(), "." | "!" | "?") {
                    should_capitalize = true;
                    words_since_punct = 0;
                } else if !is_punct(&next_word) {
                    should_capitalize = false;
                    words_since_punct += 1;
                }
                steps_since_subject += 1;
            } else {
                break;
            }
        }

        if stream {
            println!();
        }

        detokenize(&output)
    }
}

impl NudgeBot {
    fn new(precision_mode: u16) -> Self {
        Self {
            transitions: FxHashMap::default(),
            context_pairs: FxHashMap::default(),
            frequency: FxHashMap::default(),
            context_size: CONTEXT_SIZE,
            precision_mode,
            // Grammar intelligence
            capitalization: FxHashMap::default(),
            sentence_starters: FxHashMap::default(),
            after_punctuation: FxHashMap::default(),
        }
    }

    /// OPTIMIZED train with O(1) HashMap lookups, lazy normalization, and GRAMMAR AWARENESS
    /// Thread ID for progress tracking
    fn train(&mut self, text: &str, thread_id: Option<usize>) -> usize {
        let words_with_case = tokenize_with_case(text);
        self.train_tokens(words_with_case, thread_id)
    }

    fn train_filtered(
        &mut self,
        text: &str,
        banned: &FxHashSet<String>,
        thread_id: Option<usize>,
    ) -> usize {
        let words_with_case = tokenize_with_case(text);
        let filtered = filter_tokens_with_case(words_with_case, banned);
        self.train_tokens(filtered, thread_id)
    }

    fn train_tokens(&mut self, words_with_case: Vec<(String, String)>, thread_id: Option<usize>) -> usize {
        if words_with_case.is_empty() {
            return 0;
        }

        let thread_prefix = if let Some(tid) = thread_id {
            format!("[Thread {}]", tid)
        } else {
            "[◐]".to_string()
        };

        println!("{} Training on {} words...", thread_prefix, words_with_case.len());

        // OPTIMIZATION: Use temporary HashMaps during training for O(1) lookups
        // We'll convert to Vec<(String, TransitionWeight)> in finalize()
        let mut temp_transitions: FxHashMap<String, FxHashMap<String, f32>> = FxHashMap::default();
        let mut temp_context: FxHashMap<String, FxHashMap<String, u32>> = FxHashMap::default();

        // Track if we're at sentence start
        let mut at_sentence_start = true;
        let mut after_punct = false;

        // Update frequency and transitions
        for i in 0..words_with_case.len() {
            // Progress indicator for large files
            if i > 0 && i % PROGRESS_EVERY_WORDS == 0 {
                println!("{} Progress: {}/{} words...", thread_prefix, i, words_with_case.len());
            }

            let (word_lower, word_original) = &words_with_case[i];

            // GRAMMAR: Track capitalization patterns
            let is_capitalized = word_original.chars().next().map_or(false, |c| c.is_uppercase());
            let cap_entry = self.capitalization.entry(word_lower.clone()).or_insert((0, 0));
            if is_capitalized {
                cap_entry.0 += 1;  // capitalized count
            } else {
                cap_entry.1 += 1;  // lowercase count
            }

            // GRAMMAR: Track sentence starters
            if at_sentence_start && !is_punct(word_lower) {
                *self.sentence_starters.entry(word_lower.clone()).or_insert(0) += 1;
                at_sentence_start = false;
            }

            // GRAMMAR: Track words after punctuation
            if after_punct && !is_punct(word_lower) {
                *self.after_punctuation.entry(word_lower.clone()).or_insert(0) += 1;
                after_punct = false;
            }

            // Check if current word is sentence-ending punctuation
            if matches!(word_lower.as_str(), "." | "!" | "?") {
                at_sentence_start = true;
                after_punct = true;
            } else if matches!(word_lower.as_str(), "," | ";" | ":") {
                after_punct = true;
            }

            *self.frequency.entry(word_lower.clone()).or_insert(0) += 1;

            // Bigram transition (using temporary HashMap for speed)
            if i + 1 < words_with_case.len() {
                let (next_word_lower, _) = &words_with_case[i + 1];
                let entry = temp_transitions
                    .entry(word_lower.clone())
                    .or_insert_with(FxHashMap::default);

                // O(1) HashMap lookup instead of O(N) Vec::find
                let count = entry.entry(next_word_lower.clone()).or_insert(0.0);
                *count = (*count * 0.95 + 0.05).min(1.0);
            }

            // Context pairs: track words that appear together (within window)
            let max_ctx_distance = 10.min(self.context_size);
            for j in 1..=max_ctx_distance {
                if i + j < words_with_case.len() {
                    let (context_word_lower, _) = &words_with_case[i + j];
                    let decay = (1.0 / (j as f32).sqrt()) as u32;

                    let ctx_entry = temp_context
                        .entry(word_lower.clone())
                        .or_insert_with(FxHashMap::default);

                    // O(1) lookup
                    if ctx_entry.len() < CONTEXT_PAIR_LIMIT {
                        *ctx_entry.entry(context_word_lower.clone()).or_insert(0) += decay.max(1);
                    }
                }

                // Bidirectional: also track backward
                if i >= j {
                    let (context_word_lower, _) = &words_with_case[i - j];
                    let decay = (1.0 / (j as f32).sqrt()) as u32;

                    let ctx_entry = temp_context
                        .entry(word_lower.clone())
                        .or_insert_with(FxHashMap::default);

                    if ctx_entry.len() < CONTEXT_PAIR_LIMIT {
                        *ctx_entry.entry(context_word_lower.clone()).or_insert(0) += decay.max(1);
                    }
                }
            }
        }

        // Merge temporary HashMaps into final storage (but DON'T normalize yet)
        for (word, next_words) in temp_transitions {
            let entry = self.transitions.entry(word).or_insert_with(Vec::new);
            for (next_word, weight) in next_words {
                if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &next_word) {
                    let combined = pair.1.to_f32() + weight;
                    pair.1 = TransitionWeight::from_f32(combined, self.precision_mode);
                } else {
                    entry.push((next_word, TransitionWeight::from_f32(weight, self.precision_mode)));
                }
            }
        }

        // Merge context pairs
        for (word, ctx_words) in temp_context {
            let entry = self.context_pairs.entry(word).or_insert_with(Vec::new);
            for (ctx_word, count) in ctx_words {
                if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &ctx_word) {
                    pair.1 += count;
                } else if entry.len() < CONTEXT_PAIR_LIMIT {
                    entry.push((ctx_word, count));
                }
            }
        }

        println!("{} Trained with {} unique words", thread_prefix, self.frequency.len());
        words_with_case.len()
    }

    /// LAZY NORMALIZATION: Call this ONCE at the very end, not during training!
    /// This saves millions of calculations for large datasets
    fn finalize(&mut self, gpu: Option<&GpuBackend>) {
        println!("[◐] Finalizing model and normalizing weights...");
        self.normalize_transitions(gpu);
        println!("[✓] Model finalized");
    }

    fn normalize_transitions(&mut self, gpu: Option<&GpuBackend>) {
        if let Some(gpu) = gpu {
            if self.normalize_transitions_gpu(gpu) {
                return;
            }
        }
        self.normalize_transitions_cpu();
    }

    fn normalize_transitions_cpu(&mut self) {
        for transitions_list in self.transitions.values_mut() {
            let total: f32 = transitions_list
                .iter()
                .map(|(_, w)| w.to_f32())
                .sum();

            if total > 0.0 {
                for (_, w) in transitions_list.iter_mut() {
                    let normalized = w.to_f32() / total;
                    *w = TransitionWeight::from_f32(normalized, self.precision_mode);
                }
            }
        }
    }

    fn normalize_transitions_gpu(&mut self, gpu: &GpuBackend) -> bool {
        let mut transitions = std::mem::take(&mut self.transitions);
        let mut entries: Vec<(String, Vec<(String, TransitionWeight)>)> = transitions.drain().collect();
        let total_weights: usize = entries.iter().map(|(_, list)| list.len()).sum();
        if total_weights < GPU_NORMALIZE_MIN_WEIGHTS {
            self.transitions = entries.into_iter().collect();
            return false;
        }

        let mut weights: Vec<f32> = Vec::with_capacity(total_weights);
        let mut word_index: Vec<u32> = Vec::with_capacity(total_weights);
        let mut totals: Vec<f32> = Vec::with_capacity(entries.len());

        for (word_idx, (_, list)) in entries.iter().enumerate() {
            let mut total = 0.0f32;
            for (_, w) in list.iter() {
                let value = w.to_f32();
                total += value;
                weights.push(value);
                word_index.push(word_idx as u32);
            }
            totals.push(total);
        }

        if !gpu.normalize_weights(&mut weights, &word_index, &totals) {
            self.transitions = entries.into_iter().collect();
            return false;
        }

        let mut cursor = 0usize;
        for (_, list) in entries.iter_mut() {
            for (_, w) in list.iter_mut() {
                if let Some(value) = weights.get(cursor) {
                    *w = TransitionWeight::from_f32(*value, self.precision_mode);
                }
                cursor += 1;
            }
        }

        self.transitions = entries.into_iter().collect();
        true
    }

    /// Scale model contributions (used for --normalize-data)
    fn scale_model(&mut self, scale: f32) {
        if (scale - 1.0).abs() < f32::EPSILON {
            return;
        }

        // Scale frequency counts
        let mut to_remove: Vec<String> = Vec::new();
        for (word, count) in self.frequency.iter_mut() {
            let scaled = (*count as f32 * scale).round() as u32;
            *count = scaled;
            if *count == 0 {
                to_remove.push(word.clone());
            }
        }
        for word in to_remove {
            self.frequency.remove(&word);
        }

        // Scale transition weights (kept in 0..1 range)
        for transitions_list in self.transitions.values_mut() {
            for (_, w) in transitions_list.iter_mut() {
                let scaled = (w.to_f32() * scale).min(1.0);
                *w = TransitionWeight::from_f32(scaled, self.precision_mode);
            }
        }

        // Scale context pair counts
        for pairs in self.context_pairs.values_mut() {
            for (_, count) in pairs.iter_mut() {
                let scaled = (*count as f32 * scale).round() as u32;
                *count = scaled;
            }
            pairs.retain(|(_, count)| *count > 0);
        }
    }

    /// Merge another NudgeBot into this one (for parallel training)
    fn merge(&mut self, other: NudgeBot) {
        let NudgeBot {
            transitions,
            context_pairs,
            frequency,
            capitalization,
            sentence_starters,
            after_punctuation,
            ..
        } = other;

        // Merge frequencies (fast enough to keep sequential)
        for (word, count) in frequency {
            *self.frequency.entry(word).or_insert(0) += count;
        }

        // Merge transitions and context pairs in parallel when very large
        if transitions.len() >= PARALLEL_MERGE_MIN || context_pairs.len() >= PARALLEL_MERGE_MIN {
            let transitions_map = Mutex::new(std::mem::take(&mut self.transitions));
            transitions
                .into_par_iter()
                .for_each(|(word, other_list)| {
                    let mut map = transitions_map.lock().unwrap();
                    let entry = map.entry(word).or_insert_with(Vec::new);
                    for (next_word, weight) in other_list {
                        if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &next_word) {
                            let combined = pair.1.to_f32() + weight.to_f32();
                            pair.1 = TransitionWeight::from_f32(combined, self.precision_mode);
                        } else {
                            entry.push((next_word, weight));
                        }
                    }
                });
            self.transitions = transitions_map.into_inner().unwrap();

            let context_map = Mutex::new(std::mem::take(&mut self.context_pairs));
            context_pairs
                .into_par_iter()
                .for_each(|(word, other_pairs)| {
                    let mut map = context_map.lock().unwrap();
                    let entry = map.entry(word).or_insert_with(Vec::new);
                    for (ctx_word, count) in other_pairs {
                        if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &ctx_word) {
                            pair.1 += count;
                        } else if entry.len() < CONTEXT_PAIR_LIMIT {
                            entry.push((ctx_word, count));
                        }
                    }
                });
            self.context_pairs = context_map.into_inner().unwrap();
        } else {
            // Merge transitions
            for (word, other_list) in transitions {
                let entry = self.transitions.entry(word).or_insert_with(Vec::new);
                for (next_word, weight) in other_list {
                    if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &next_word) {
                        let combined = pair.1.to_f32() + weight.to_f32();
                        pair.1 = TransitionWeight::from_f32(combined, self.precision_mode);
                    } else {
                        entry.push((next_word, weight));
                    }
                }
            }

            // Merge context pairs
            for (word, other_pairs) in context_pairs {
                let entry = self.context_pairs.entry(word).or_insert_with(Vec::new);
                for (ctx_word, count) in other_pairs {
                    if let Some(pair) = entry.iter_mut().find(|(w, _)| w == &ctx_word) {
                        pair.1 += count;
                    } else if entry.len() < CONTEXT_PAIR_LIMIT {
                        entry.push((ctx_word, count));
                    }
                }
            }
        }

        // GRAMMAR: Merge capitalization patterns
        for (word, (cap_count, lower_count)) in capitalization {
            let entry = self.capitalization.entry(word).or_insert((0, 0));
            entry.0 += cap_count;
            entry.1 += lower_count;
        }

        // GRAMMAR: Merge sentence starters
        for (word, count) in sentence_starters {
            *self.sentence_starters.entry(word).or_insert(0) += count;
        }

        // GRAMMAR: Merge after-punctuation patterns
        for (word, count) in after_punctuation {
            *self.after_punctuation.entry(word).or_insert(0) += count;
        }
    }

    /// Renormalize all transitions after merging (called once after parallel merge)
    fn renormalize(&mut self, gpu: Option<&GpuBackend>) {
        self.normalize_transitions(gpu);
    }

    /// Apply advanced context nudges with bidirectional awareness
    fn apply_context_nudges(
        &self,
        word: &str,
        context_before: &VecDeque<String>,
        context_after: &VecDeque<String>,
        subject_weights: &FxHashMap<String, f32>,
    ) -> Vec<(String, f32)> {
        if let Some(transitions) = self.transitions.get(word) {
            let mut nudged: Vec<(String, f32)> = transitions
                .iter()
                .map(|(w, q)| (w.clone(), q.to_f32()))
                .collect();

            // Penalize words that already appear in recent context (avoid repetition)
            // Use semantic clustering to penalize similar words too
            let all_context: Vec<String> = context_before.iter().chain(context_after.iter()).cloned().collect();
            for (next_word, weight) in nudged.iter_mut() {
                // AGGRESSIVE exact match penalty with sliding window
                // Recent words (last 20) get MUCH higher penalty
                let recent_window = all_context.iter().rev().take(20);
                let older_context = all_context.iter().rev().skip(20);

                let recent_exact = recent_window.clone().filter(|w| *w == next_word).count() as f32;
                let older_exact = older_context.clone().filter(|w| *w == next_word).count() as f32;

                // Recent repetitions are FAR worse (5x penalty)
                let exact_count = recent_exact * 5.0 + older_exact;

                // IMPROVED semantic similarity penalty
                let mut recent_semantic = 0.0;
                let mut older_semantic = 0.0;

                for ctx_word in recent_window {
                    let similarity = semantic_similarity(next_word, ctx_word);
                    if similarity >= 0.3 {  // Lower threshold
                        recent_semantic += similarity;
                    }
                }

                for ctx_word in older_context {
                    let similarity = semantic_similarity(next_word, ctx_word);
                    if similarity >= 0.3 {
                        older_semantic += similarity;
                    }
                }

                let semantic_count = recent_semantic * 3.0 + older_semantic;

                // MUCH higher combined penalty
                let total_penalty = exact_count * 0.9 + semantic_count * 0.6;
                if total_penalty > 0.0 {
                    // VERY aggressive: exponential penalty for severe repetition
                    *weight /= (1.0 + total_penalty).powf(1.5);
                }

                // Extra anti-repeat: penalize recent exact matches harder
                if let Some(last_token) = context_before.back() {
                    if last_token == next_word {
                        *weight *= 0.05;
                    }
                }
                if context_before.iter().rev().take(8).any(|w| w == next_word) {
                    *weight *= 0.2;
                }
            }

            // Apply nudges from context BEFORE (backward) - only recent context for speed
            for (distance, ctx_word) in context_before.iter().rev().take(15).enumerate() {
                let distance_factor = 1.0 / (1.0 + distance as f32 * 0.3); // Gradual decay

                for (next_word, base_weight) in nudged.iter_mut() {
                    // Check if context word leads to this next word
                    if let Some(ctx_transitions) = self.transitions.get(ctx_word) {
                        if let Some((_, ctx_weight)) = ctx_transitions
                            .iter()
                            .find(|(w, _)| w == next_word)
                        {
                            let support = ctx_weight.to_f32();
                            let nudge = support * distance_factor * 0.11; // 12% max influence
                            *base_weight = (*base_weight + nudge).min(1.0);
                        }
                    }

                    // Also check context pairs for associative nudging (skip if expensive)
                    if let Some(pairs) = self.context_pairs.get(ctx_word) {
                        if pairs.len() < CONTEXT_PAIR_SEARCH_LIMIT {
                            if let Some((_, count)) = pairs.iter().find(|(w, _)| w == next_word) {
                                let pair_strength = (*count as f32 / 100.0).min(1.0);
                                let nudge = pair_strength * distance_factor * 0.08;
                                *base_weight = (*base_weight + nudge).min(1.0);
                            }
                        }
                    }
                }
            }

            // Apply nudges from context AFTER (forward) - only nearby context
            for (distance, ctx_word) in context_after.iter().take(15).enumerate() {
                let distance_factor = 1.0 / (1.0 + distance as f32 * 0.5); // Faster decay for future

                for (next_word, base_weight) in nudged.iter_mut() {
                    if let Some(pairs) = self.context_pairs.get(ctx_word) {
                        if pairs.len() < CONTEXT_PAIR_FORWARD_SEARCH_LIMIT {
                            if let Some((_, count)) = pairs.iter().find(|(w, _)| w == next_word) {
                                let pair_strength = (*count as f32 / 100.0).min(1.0);
                                let nudge = pair_strength * distance_factor * 0.05; // 5% max for future
                                *base_weight = (*base_weight + nudge).min(1.0);
                            }
                        }
                    }
                }
            }

            // Subject bias: keep replies on-topic (INCREASED from 12% to 30%)
            for (next_word, base_weight) in nudged.iter_mut() {
                if let Some(score) = subject_weights.get(next_word) {
                    let nudge = (score * 0.30).min(0.30);  // Increased influence
                    *base_weight = (*base_weight + nudge).min(1.0);
                }
            }

            // Normalize
            let total: f32 = nudged.iter().map(|(_, w)| w).sum();
            if total > 0.0 {
                for (_, weight) in nudged.iter_mut() {
                    *weight /= total;
                }
            }

            nudged
        } else {
            vec![]
        }
    }

    /// Generate with advanced context awareness
    fn generate(&self, count: usize, seed_word: Option<&str>) -> String {
        let context_before: VecDeque<String> = VecDeque::new();
        let style_counts: FxHashMap<String, u32> = FxHashMap::default();
        self.generate_with_context(count, seed_word, &context_before, &style_counts)
    }

    /// Pure first-order Markov generation (no context, subject tracking, or user weighting)
    /// Just: given current word, pick next word based on transition probabilities
    fn pure_markov_generate(&self, count: usize, seed_word: Option<&str>) -> String {
        let mut rng = rand::thread_rng();
        let mut output: Vec<String> = Vec::new();

        // Pick starting word
        let mut current = if let Some(seed) = seed_word {
            seed.to_lowercase()
        } else {
            // Random start from frequent words
            let freq_words: Vec<_> = self.frequency.iter()
                .filter(|(_, count)| **count > 3)
                .collect();
            if freq_words.is_empty() {
                return String::new();
            }
            freq_words[rng.gen_range(0..freq_words.len())].0.clone()
        };

        output.push(current.clone());

        // Pure Markov chain: only use transitions
        for _ in 0..count {
            if let Some(next_words) = self.transitions.get(&current) {
                if next_words.is_empty() {
                    break;
                }

                // Simple weighted sampling based on transition weights
                let weights: Vec<f32> = next_words.iter()
                    .map(|(_, w)| w.to_f32())
                    .collect();

                if let Ok(dist) = WeightedIndex::new(&weights) {
                    let idx = dist.sample(&mut rng);
                    current = next_words[idx].0.clone();
                    output.push(current.clone());
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        output.join(" ")
    }

    fn generate_with_context(
        &self,
        count: usize,
        seed_word: Option<&str>,
        context_before: &VecDeque<String>,
        style_counts: &FxHashMap<String, u32>,
    ) -> String {
        self.generate_with_context_and_user_boost(count, seed_word, context_before, style_counts, 0, false, None)
    }

    fn generate_with_context_and_user_boost(
        &self,
        count: usize,
        seed_word: Option<&str>,
        context_before: &VecDeque<String>,
        style_counts: &FxHashMap<String, u32>,
        user_token_count: usize,  // Number of recent tokens from user (for weighting)
        stream: bool,
        pass_text: Option<&str>,
    ) -> String {
        let mut rng = rand::thread_rng();
        let mut output: Vec<String> = Vec::new();
        let mut local_context = context_before.clone();
        let context_after: VecDeque<String> = VecDeque::with_capacity(self.context_size);
        let mut stream_first = true;
        let mut stream_last: Option<String> = None;
        let mut stream_in_quote = false;
        // Pick starting word (keep lowercase for lookups)
        let mut current_word = if let Some(pass) = pass_text {
            let pass_tokens = tokenize_with_case(pass);
            for (lower, original) in &pass_tokens {
                output.push(original.clone());
                if stream {
                    stream_token(original, &mut stream_first, &mut stream_last, &mut stream_in_quote);
                }
                update_context(&mut local_context, &[lower.clone()], self.context_size);
            }
            if let Some((last_lower, _)) = pass_tokens.last() {
                last_lower.clone()
            } else if let Some(seed) = seed_word {
                seed.to_lowercase()
            } else {
                self.transitions
                    .keys()
                    .choose(&mut rng)
                    .cloned()
                    .unwrap_or_else(|| "the".to_string())
            }
        } else if let Some(seed) = seed_word {
            seed.to_lowercase()
        } else {
            self.transitions
                .keys()
                .choose(&mut rng)
                .cloned()
                .unwrap_or_else(|| "the".to_string())
        };

        if pass_text.is_none() {
            // Output formatted start word but keep lowercase for transitions
            let start_word = self.apply_grammar(&current_word, true, 0);
            output.push(start_word.clone());
            if stream {
                stream_token(&start_word, &mut stream_first, &mut stream_last, &mut stream_in_quote);
            }
            update_context(&mut local_context, &[current_word.clone()], self.context_size);
        }

        // GRAMMAR: Track if we need to capitalize (start of sentence)
        let mut should_capitalize = match local_context.back() {
            Some(tok) if matches!(tok.as_str(), "." | "!" | "?") => true,
            _ => false,
        };
        let mut words_since_punct = 0;
        if !local_context.is_empty() {
            for tok in local_context.iter().rev() {
                if matches!(tok.as_str(), "." | "!" | "?") {
                    break;
                }
                if !is_punct(tok) {
                    words_since_punct += 1;
                }
            }
        }

        let mut subject_weights = if user_token_count > 0 {
            build_subject_weights_with_user_boost(&local_context, user_token_count)
        } else {
            build_subject_weights(&local_context)
        };
        let mut steps_since_subject = 0usize;

        while output.len() < count {
            if steps_since_subject >= 5 {
                subject_weights = if user_token_count > 0 {
                    build_subject_weights_with_user_boost(&local_context, user_token_count)
                } else {
                    build_subject_weights(&local_context)
                };
                steps_since_subject = 0;
            }

            // Get nudged transitions with bidirectional context
            let candidates = self.apply_context_nudges(
                &current_word,
                &local_context,
                &context_after,
                &subject_weights,
            );

            if candidates.is_empty() {
                break;
            }

            let weights: Vec<f32> = candidates.iter().map(|(_, w)| w).copied().collect();
            if let Some(idx) = weighted_choice(&weights, &mut rng) {
                let next_word = candidates[idx].0.clone();

                // GRAMMAR: Apply learned capitalization patterns
                let formatted_word = self.apply_grammar(&next_word, should_capitalize, words_since_punct);

                output.push(formatted_word.clone());
                if stream {
                    stream_token(&formatted_word, &mut stream_first, &mut stream_last, &mut stream_in_quote);
                }
                update_context(&mut local_context, &[next_word.clone()], self.context_size);
                current_word = next_word.clone();

                // GRAMMAR: Update sentence state
                if matches!(next_word.as_str(), "." | "!" | "?") {
                    should_capitalize = true;
                    words_since_punct = 0;
                } else {
                    if !is_punct(&next_word) {
                        should_capitalize = false;
                        words_since_punct += 1;
                    }
                }

                if let Some(style_token) = pick_style_token(style_counts, &mut rng) {
                    output.push(style_token);
                }
                steps_since_subject += 1;
            } else {
                break;
            }
        }

        if stream {
            println!();
        }

        detokenize(&output)
    }

    /// GRAMMAR INTELLIGENCE: Apply learned capitalization patterns
    fn apply_grammar(&self, word: &str, should_capitalize: bool, words_since_punct: usize) -> String {
        // Don't modify punctuation
        if is_punct(word) {
            return word.to_string();
        }

        // Check our learned capitalization patterns
        if let Some((cap_count, lower_count)) = self.capitalization.get(word) {
            let total = cap_count + lower_count;
            if total == 0 {
                return word.to_string();
            }

            let cap_ratio = *cap_count as f32 / total as f32;

            // Strong capitalization preference (>70% of the time it's capitalized)
            let usually_capitalized = cap_ratio > 0.7;

            // Proper nouns, names, etc.
            if usually_capitalized {
                return capitalize_first(word);
            }

            // Sentence start
            if should_capitalize {
                return capitalize_first(word);
            }

            // Check if it's a common sentence starter
            if let Some(_count) = self.sentence_starters.get(word) {
                if words_since_punct == 0 {
                    return capitalize_first(word);
                }
            }
        }

        // Default: keep lowercase
        word.to_string()
    }

    /// Save with compression
    fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::serialize(self)?;
        let compressed = zstd::encode_all(&serialized[..], 10)?; // Higher compression
        let size = compressed.len();
        fs::write(path, compressed)?;
        println!("[✓] Model saved ({} bytes, compressed)", size);
        Ok(())
    }

    /// Load from compressed storage
    fn load(path: &str) -> Result<Self> {
        let compressed = fs::read(path)?;
        let serialized = zstd::decode_all(&compressed[..])?;
        let bot = bincode::deserialize(&serialized)?;
        println!("[✓] Model loaded");
        Ok(bot)
    }
}

/// Weighted random choice using O(1) sampling (Walker's Alias Method via WeightedIndex)
fn weighted_choice(weights: &[f32], rng: &mut impl Rng) -> Option<usize> {
    if weights.is_empty() {
        return None;
    }

    // WeightedIndex handles normalization and builds the alias table O(n)
    // but sampling is O(1) per call
    match WeightedIndex::new(weights) {
        Ok(dist) => Some(dist.sample(rng)),
        Err(_) => None,
    }
}

fn parse_pass_arg(args: &[String]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--pass" {
            if i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
        } else if let Some(rest) = arg.strip_prefix("--pass=") {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

fn parse_gpu_arg(args: &[String]) -> String {
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--gpu" {
            if let Some(value) = args.get(i + 1) {
                return value.to_ascii_lowercase();
            }
        } else if let Some(rest) = arg.strip_prefix("--gpu=") {
            return rest.to_ascii_lowercase();
        }
        i += 1;
    }
    "auto".to_string()
}

fn stream_token(
    token: &str,
    is_first: &mut bool,
    last_token: &mut Option<String>,
    in_quote: &mut bool,
) {
    if is_quote_token(token) {
        if *in_quote {
            print!("{}", token);
            *in_quote = false;
        } else {
            if !*is_first {
                print!(" ");
            }
            print!("{}", token);
            *in_quote = true;
        }
        *is_first = false;
        *last_token = Some(token.to_string());
        let _ = io::stdout().flush();
        return;
    }

    if is_punct(token) {
        if let Some(prev) = last_token.as_deref() {
            if is_punct(prev) {
                if prev == token {
                    print!("{}", token);
                    *last_token = Some(token.to_string());
                }
                let _ = io::stdout().flush();
                return;
            }
        }
        print!("{}", token);
        *is_first = false;
        *last_token = Some(token.to_string());
        let _ = io::stdout().flush();
        return;
    }

    let needs_space = !*is_first && !(*in_quote && matches!(last_token.as_deref(), Some("\"")));
    if needs_space {
        print!(" ");
    }
    print!("{}", token);
    *is_first = false;
    *last_token = Some(token.to_string());
    let _ = io::stdout().flush();
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <train|generate|talk|train-fragment|merge-fragments|freeze> [options]", args[0]);
        eprintln!("Train on datas/ folder: {} train [--bits=8|64|128|256] [--normalize-data]", args[0]);
        eprintln!("Generate: {} generate {} [--bits=8|64|128|256] [--pure] [--pass <text>] [--stream]", args[0], DEFAULT_GENERATE_COUNT);
        eprintln!("Talk: {} talk {} [--bits=8|64|128|256]", args[0], DEFAULT_TALK_COUNT);
        eprintln!("Train fragment: {} train-fragment <file> [--bits=8|64|128|256]", args[0]);
        eprintln!("Merge fragments: {} merge-fragments [--bits=8|64|128|256]", args[0]);
        eprintln!("Freeze: {} freeze [--bits=8|64|128|256]", args[0]);
        eprintln!("Default: 8-bit quantized mode (fastest)");
        eprintln!("--pure: Pure first-order Markov (no context/subject tracking)");
        eprintln!("--normalize-data: equalize per-file influence during training");
        eprintln!("--pass: provide a starting prompt for generation");
        eprintln!("--stream: print tokens as they are generated");
        eprintln!("--gpu: auto|nvidia|amd|intel|off (GPU normalize when available)");
        std::process::exit(1);
    }

    // Check for precision mode flag
    let precision_mode: u16 = if args.iter().any(|arg| arg.contains("--bits=256")) {
        println!("[◐] Using TRUE 128-bit (u128) mode");
        256
    } else if args.iter().any(|arg| arg.contains("--bits=128")) {
        println!("[◐] Using 128-bit (f64) mode");
        128
    } else if args.iter().any(|arg| arg.contains("--bits=64")) {
        println!("[◐] Using 64-bit (f32) mode");
        64
    } else {
        println!("[◐] Using 8-bit (quantized) mode");
        8
    };

    // Check for pure Markov mode
    let pure_markov = args.iter().any(|arg| arg == "--pure");
    if pure_markov {
        println!("[◐] Pure Markov mode: first-order only, no context");
    }

    // Normalize data so each file contributes equally
    let normalize_data = args.iter().any(|arg| arg == "--normalize-data");
    if normalize_data {
        println!("[◐] Normalize data: equalize per-file influence");
    }

    let model_path = "nudge_bot_model.bin";
    let frozen_path = "nudge_bot_model.frozen";
    let pass_text = parse_pass_arg(&args);
    let stream = args.iter().any(|arg| arg == "--stream");
    let command = &args[1];
    let banned_words = Arc::new(load_nono_words("nono.txt"));
    let gpu_preference = parse_gpu_arg(&args);
    let needs_training = matches!(command.as_str(), "train" | "t" | "train-fragment" | "tf" | "merge-fragments" | "mf");
    let gpu_backend = if needs_training && gpu_preference != "off" && gpu_preference != "cpu" {
        GpuBackend::init(&gpu_preference)
    } else {
        None
    };
    if needs_training {
        if gpu_backend.is_some() {
            println!("[◐] GPU normalize enabled: {}", gpu_preference);
        } else if gpu_preference != "off" && gpu_preference != "cpu" {
            println!("[◐] GPU not available - using CPU");
        }
    }

    match command.as_str() {
        "train" | "t" => {
            println!("[◐] PARALLEL training on files from datas/ folder...");

            // Read all supported files from datas/ folder (parallel discovery)
            let data_dir = Path::new("datas");
            if !data_dir.exists() {
                anyhow::bail!("datas/ folder not found");
            }

            let ignore_dir = data_dir.join("ignores");
            let files: Vec<_> = fs::read_dir(data_dir)?
                .par_bridge()
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.starts_with(&ignore_dir) {
                        return None;
                    }
                    if path.is_file() && is_supported_training_file(&path) {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect();

            if files.is_empty() {
                anyhow::bail!("No .txt files found in datas/ folder");
            }

            println!("[◐] Found {} files - training on {} cores...", files.len(), rayon::current_num_threads());

            // Precompute per-file token counts if normalization is enabled
            let file_stats: Vec<(std::path::PathBuf, usize)> = if normalize_data {
                files.par_iter().map(|path| {
                    let texts = read_training_texts(path).unwrap_or_default();
                    if texts.is_empty() {
                        return (path.clone(), 0);
                    }

                    let count: usize = texts
                        .iter()
                        .map(|text| {
                            let purified = purify_text(text);
                            count_tokens_filtered(&purified, banned_words.as_ref())
                        })
                        .sum();
                    (path.clone(), count)
                }).collect()
            } else {
                files.into_iter().map(|path| (path, 0)).collect()
            };

            let avg_tokens: f32 = if normalize_data {
                let total: usize = file_stats.iter().map(|(_, count)| *count).sum();
                if total == 0 {
                    0.0
                } else {
                    total as f32 / file_stats.len() as f32
                }
            } else {
                0.0
            };

            // Thread counter for worker IDs
            let thread_counter = Arc::new(AtomicUsize::new(0));
            let worker_count = rayon::current_num_threads().max(2);
            let merge_workers = (worker_count / 2).max(1);
            println!("[◐] Training workers: {} | Merge workers: {}", worker_count, merge_workers);

            let (file_tx, file_rx) = unbounded::<(std::path::PathBuf, usize)>();
            for item in file_stats {
                let _ = file_tx.send(item);
            }
            drop(file_tx);

            let (bot_tx, bot_rx) = unbounded::<NudgeBot>();
            let (merge_tx, merge_rx) = unbounded::<NudgeBot>();

            let mut worker_handles = Vec::new();
            for _ in 0..worker_count {
                let rx = file_rx.clone();
                let tx = bot_tx.clone();
                let thread_counter = Arc::clone(&thread_counter);
                let banned_words = Arc::clone(&banned_words);
                let handle = thread::spawn(move || {
                    let worker_id = thread_counter.fetch_add(1, Ordering::Relaxed) + 1;
                    while let Ok((path, token_count)) = rx.recv() {
                        let mut local_bot = NudgeBot::new(precision_mode);
                        if let Ok(texts) = read_training_texts(&path) {
                            if texts.is_empty() {
                                let _ = tx.send(local_bot);
                                continue;
                            }

                            println!("[Thread {}] Processing: {}", worker_id, path.display());

                            let scale = if normalize_data && token_count > 0 {
                                (avg_tokens / token_count as f32).min(1.0)
                            } else {
                                1.0
                            };

                            for text in texts {
                                // PURIFY: Remove URLs, markdown, forewords, navigation, etc.
                                let purified = purify_text(&text);
                                if purified.is_empty() {
                                    continue;
                                }

                                // Process giant files with parallel chunks for HUGE files (1M+ words)
                                if purified.len() > LARGE_FILE_BYTES {
                                    println!("[Thread {}] Large file detected ({} bytes) - chunking...", worker_id, purified.len());

                                    // Split into lines and process in parallel batches
                                    let lines: Vec<&str> = purified.lines().collect();
                                    let threads = rayon::current_num_threads().max(2);
                                    let target_chunks = threads * 4;
                                    let mut chunk_size = CHUNK_LINES;
                                    if lines.len() > chunk_size {
                                        let dynamic = (lines.len() / target_chunks).max(1);
                                        chunk_size = chunk_size.max(dynamic);
                                    }

                                    // Create sub-bots for each chunk
                                    let chunk_bots: Vec<NudgeBot> = lines.par_chunks(chunk_size)
                                        .enumerate()
                                        .map(|(chunk_idx, chunk_lines)| {
                                            let mut chunk_bot = NudgeBot::new(precision_mode);
                                            let chunk_text = chunk_lines.join(" ");
                                            let _ = chunk_idx;
                                            let _ = chunk_bot.train_filtered(&chunk_text, banned_words.as_ref(), Some(worker_id));
                                            chunk_bot
                                        })
                                        .collect();

                                    // Merge all chunk bots into local bot
                                    println!("[Thread {}] Merging {} chunks...", worker_id, chunk_bots.len());
                                    let merged_chunks = chunk_bots
                                        .into_par_iter()
                                        .reduce(|| NudgeBot::new(precision_mode), |mut bot_a, bot_b| {
                                            bot_a.merge(bot_b);
                                            bot_a
                                        });
                                    local_bot.merge(merged_chunks);
                                } else {
                                    let _ = local_bot.train_filtered(&purified, banned_words.as_ref(), Some(worker_id));
                                }
                            }

                            if normalize_data {
                                local_bot.scale_model(scale);
                            }
                            println!("[Thread {}] Complete: {}", worker_id, path.display());
                        }
                        let _ = tx.send(local_bot);
                    }
                });
                worker_handles.push(handle);
            }
            drop(bot_tx);

            let mut merge_handles = Vec::new();
            for _ in 0..merge_workers {
                let rx = bot_rx.clone();
                let tx = merge_tx.clone();
                let handle = thread::spawn(move || {
                    let mut merged = NudgeBot::new(precision_mode);
                    while let Ok(bot) = rx.recv() {
                        merged.merge(bot);
                    }
                    let _ = tx.send(merged);
                });
                merge_handles.push(handle);
            }
            drop(merge_tx);
            drop(bot_rx);

            for handle in worker_handles {
                let _ = handle.join();
            }

            let mut final_bot = NudgeBot::new(precision_mode);
            for _ in 0..merge_workers {
                if let Ok(partial) = merge_rx.recv() {
                    final_bot.merge(partial);
                }
            }
            for handle in merge_handles {
                let _ = handle.join();
            }

            // FINALIZE: Lazy normalization happens ONCE at the very end
            println!("[◐] All threads complete - finalizing model...");
            final_bot.finalize(gpu_backend.as_ref());

            final_bot.save(model_path)?;
            println!("[✓] Parallel training complete!");
        }
        "generate" | "g" => {
            let count = args
                .get(2)
                .and_then(|c| c.parse().ok())
                .unwrap_or(DEFAULT_GENERATE_COUNT);

            let output = if pure_markov {
                let bot = NudgeBot::load(model_path)?;
                bot.pure_markov_generate(count, None)
            } else if fs::metadata(frozen_path).is_ok() {
                let frozen = FrozenModel::load(frozen_path)?;
                frozen.generate_with_pass(count, pass_text.as_deref(), stream)
            } else {
                let bot = NudgeBot::load(model_path)?;
                bot.generate_with_context_and_user_boost(count, None, &VecDeque::new(), &FxHashMap::default(), 0, stream, pass_text.as_deref())
            };
            if !stream {
                println!("{}", output);
            }
        }
        "talk" | "chat" => {
            let count = args
                .get(2)
                .and_then(|c| c.parse().ok())
                .unwrap_or(DEFAULT_TALK_COUNT);

            let bot = NudgeBot::load(model_path)?;
            let mut context: VecDeque<String> = VecDeque::with_capacity(bot.context_size);
            let mut style_counts: FxHashMap<String, u32> = FxHashMap::default();
            // Cross-turn memory: keeps track of all topics discussed in conversation
            let mut conversation_history: VecDeque<String> = VecDeque::with_capacity(CONVERSATION_HISTORY_CAP);

            println!("Talk mode. Type /exit to quit.");

            loop {
                print!("> ");
                io::stdout().flush()?;

                let mut line = String::new();
                if io::stdin().read_line(&mut line)? == 0 {
                    break;
                }

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "/exit" || trimmed == "/quit" {
                    break;
                }

                let tokens = tokenize(trimmed);
                let user_token_count = tokens.len();  // Track how many tokens user sent
                update_style_counts(&mut style_counts, &tokens);
                update_context(&mut context, &tokens, bot.context_size);
                // Add to long-term conversation history
                for token in &tokens {
                    conversation_history.push_back(token.clone());
                    if conversation_history.len() > CONVERSATION_HISTORY_CAP {
                        conversation_history.pop_front();
                    }
                }

                // Generate reply with user message weighted higher (2.5x boost)
                let reply = bot.generate_with_context_and_user_boost(count, None, &context, &style_counts, user_token_count, false, None);
                println!("{}", reply);

                let reply_tokens = tokenize(&reply);
                update_context(&mut context, &reply_tokens, bot.context_size);
                // Also add bot response to conversation history
                for token in &reply_tokens {
                    conversation_history.push_back(token.clone());
                    if conversation_history.len() > CONVERSATION_HISTORY_CAP {
                        conversation_history.pop_front();
                    }
                }
            }
        }
        "train-fragment" | "tf" => {
            // Train on a single file and save as a fragment
            if args.len() < 3 {
                anyhow::bail!("Usage: {} train-fragment <filepath>", args[0]);
            }

            let file_path = Path::new(&args[2]);
            if !file_path.exists() {
                anyhow::bail!("File not found: {}", file_path.display());
            }

            println!("[◐] Training fragment from: {}", file_path.display());
            let mut bot = NudgeBot::new(precision_mode);
            let texts = read_training_texts(file_path)?;
            if texts.is_empty() {
                anyhow::bail!("No readable text found in {}", file_path.display());
            }

            // PURIFY: Remove URLs, markdown, forewords, navigation, etc.
            println!("[◐] Purifying training data...");
            for text in texts {
                let purified = purify_text(&text);
                if purified.is_empty() {
                    continue;
                }
                let _ = bot.train_filtered(&purified, banned_words.as_ref(), None);  // None = no thread ID for single-file training
            }

            // FINALIZE before saving
            bot.finalize(gpu_backend.as_ref());

            // Save as .fragment file in fragments/ directory
            fs::create_dir_all("fragments")?;
            let fragment_name = file_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("fragment");
            let fragment_path = format!("fragments/{}.fragment", fragment_name);
            bot.save(&fragment_path)?;
            println!("[✓] Fragment saved: {}", fragment_path);
        }
        "merge-fragments" | "mf" => {
            // Merge all .fragment files from fragments/ directory
            let fragments_dir = Path::new("fragments");
            if !fragments_dir.exists() {
                anyhow::bail!("fragments/ folder not found - train some fragments first!");
            }

            let fragments: Vec<_> = fs::read_dir(fragments_dir)?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("fragment") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect();

            if fragments.is_empty() {
                anyhow::bail!("No .fragment files found in fragments/");
            }

            println!("[◐] Found {} fragments - merging...", fragments.len());

            // Load first fragment as base
            let mut merged = NudgeBot::load(fragments[0].to_str().unwrap())?;
            println!("[◐] Base: {}", fragments[0].display());

            // Merge remaining fragments
            for fragment_path in &fragments[1..] {
                println!("[◐] Merging: {}", fragment_path.display());
                let fragment = NudgeBot::load(fragment_path.to_str().unwrap())?;
                merged.merge(fragment);
            }

            // FINALIZE after merging all fragments
            println!("[◐] Finalizing merged model...");
            merged.finalize(gpu_backend.as_ref());

            merged.save(model_path)?;
            println!("[✓] Merged {} fragments into {}", fragments.len(), model_path);
        }
        "freeze" | "fz" => {
            // Build a frozen model with index-based storage (perfect-hash ready)
            let bot = NudgeBot::load(model_path)?;
            let frozen = FrozenModel::from_bot(&bot);
            frozen.save(frozen_path)?;
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Available: train, generate, talk, train-fragment, merge-fragments, freeze");
            std::process::exit(1);
        }
    }

    Ok(())
}
