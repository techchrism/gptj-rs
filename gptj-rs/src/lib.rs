mod ggml;

use std::{
    collections::HashMap,
    fmt::Display,
    io::{BufRead, Read},
    path::{Path, PathBuf},
};

use thiserror::Error;
use fancy_regex::Regex;

use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hyperparameters {
    n_vocab: i32,
    n_ctx: i32,
    n_embd: i32,
    n_head: i32,
    n_layer: i32,
    n_rot: i32,
    f16_: i32,
}

struct Layer {
    // normalization
    ln_1_g: ggml::Tensor,
    ln_1_b: ggml::Tensor,

    // attention
    c_attn_q_proj_w: ggml::Tensor,
    c_attn_k_proj_w: ggml::Tensor,
    c_attn_v_proj_w: ggml::Tensor,

    c_attn_proj_w: ggml::Tensor,

    // ff
    c_mlp_fc_w: ggml::Tensor,
    c_mlp_fc_b: ggml::Tensor,
    c_mlp_proj_w_trans: ggml::Tensor,
    c_mlp_proj_b: ggml::Tensor,
}

pub struct Model {
    hparams: Hyperparameters,

    // normalization
    ln_f_g: ggml::Tensor,
    ln_f_b: ggml::Tensor,

    tok_embeddings: ggml::Tensor, // wte

    lmh_g: ggml::Tensor, // language model head
    lmh_b: ggml::Tensor, // language model bias

    layers: Vec<Layer>,

    memory_k: ggml::Tensor,
    memory_v: ggml::Tensor,

    tensors: HashMap<String, ggml::Tensor>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

pub struct InferenceParameters {
    pub n_threads: i32,
    pub n_predict: usize,
    pub n_batch: usize,
    pub repeat_last_n: usize,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub temp: f32,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_predict: 128,
            n_batch: 8,
            repeat_last_n: 64,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temp: 0.80,
        }
    }
}

type TokenId = i32;
type Token = String;

#[derive(Default)]
pub struct Vocabulary {
    /// Maps every integer (index) token id to its corresponding string
    mapping: Vec<Token>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputToken<'a> {
    Token(&'a str),
    EndOfText,
}
impl Display for OutputToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                OutputToken::Token(t) => *t,
                OutputToken::EndOfText => "[end of text]",
            }
        )
    }
}

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a> {
    HyperParamsLoaded(&'a Hyperparameters),
    BadToken {
        index: usize,
    },
    ContextSize {
        bytes: usize,
    },
    MemorySize {
        bytes: usize,
        n_mem: usize,
    },
    TensorLoading {
        current_tensor: usize,
        tensor_count: usize,
    },
    TensorLoaded {
        current_tensor: usize,
        tensor_count: usize,
    }
}

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("could not open file {path:?}")]
    OpenFileFailed {
        source: std::io::Error,
        path: PathBuf,
    },
    #[error("unable to read exactly {bytes} bytes")]
    ReadExactFailed {
        source: std::io::Error,
        bytes: usize,
    },
    #[error("non-specific I/O error")]
    IO(#[from] std::io::Error),

    #[error("could not convert bytes to a UTF-8 string")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),

    #[error("invalid magic number for {path:?}")]
    InvalidMagic { path: PathBuf },
    #[error("invalid vocab length in model (expected {n_vocab}, got {vocab_len})")]
    VocabLengthMismatch { n_vocab: i32, vocab_len: i32 },
    #[error("invalid value {value} for `f16` in hyperparameters")]
    HyperparametersF16Invalid { value: i32 },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    UnknownTensor { tensor_name: String, path: PathBuf },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    TensorWrongSize { tensor_name: String, path: PathBuf },
    #[error("invalid ftype {ftype} in {path:?}")]
    InvalidFtype { ftype: i32, path: PathBuf },
}

impl Model {
    pub fn load(
        path: impl AsRef<Path>,
        load_progress_callback: impl Fn(LoadProgress),
    ) -> Result<(Model, Vocabulary), LoadError> {
        use std::fs::File;
        use std::io::BufReader;

        let path = path.as_ref();

        let mut reader =
            BufReader::new(File::open(path).map_err(|e| LoadError::OpenFileFailed {
                source: e,
                path: path.to_owned(),
            })?);

        /// Helper function. Reads an int from the buffer and returns it.
        fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
            let mut bytes = [0u8; 4];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: bytes.len(),
                })?;
            Ok(i32::from_le_bytes(bytes))
        }

        /// Helper function. Reads a string from the buffer and returns it.
        fn read_string(reader: &mut BufReader<File>, len: usize) -> Result<String, LoadError> {
            let mut buf = vec![0; len];
            reader
                .read_exact(&mut buf)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: buf.len(),
                })?;
            let s = String::from_utf8(buf)?;
            Ok(s)
        }

        // Verify magic
        {
            let magic = read_i32(&mut reader)?;
            if magic != 0x67676d6c {
                return Err(LoadError::InvalidMagic {
                    path: path.to_owned(),
                });
            }
        }

        // =================
        // Load hyper params
        // =================

        // NOTE: Field order matters! Data is laid out in the file exactly
        // in this order.
        let hparams = Hyperparameters {
            n_vocab: read_i32(&mut reader)?,
            n_ctx: read_i32(&mut reader)?,
            n_embd: read_i32(&mut reader)?,
            n_head: read_i32(&mut reader)?,
            n_layer: read_i32(&mut reader)?,
            n_rot: read_i32(&mut reader)?,
            f16_: read_i32(&mut reader)?,
        };

        load_progress_callback(LoadProgress::HyperParamsLoaded(&hparams));

        // ===============
        // Load vocabulary
        // ===============
        let mut vocab = Vocabulary::default();
        let vocab_len = read_i32(&mut reader)?;
        if vocab_len != hparams.n_vocab {
            return Err(LoadError::VocabLengthMismatch { n_vocab: hparams.n_vocab, vocab_len });
        }
        for i in 0..hparams.n_vocab {
            let len = read_i32(&mut reader)?;
            if let Ok(word) = read_string(&mut reader, len as usize) {
                vocab.mapping.push(word);
            } else {
                /*load_progress_callback(LoadProgress::BadToken {
                    index: i.try_into()?,
                });*/
                vocab.mapping.push("�".to_string());
            }
        }

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        let wtype = match hparams.f16_ {
            0 => ggml::TYPE_F32,
            1 => ggml::TYPE_F16,
            invalid => return Err(LoadError::HyperparametersF16Invalid { value: invalid }),
        };

        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_ctx = hparams.n_ctx;
        let n_vocab = hparams.n_vocab;

        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let n_embd = n_embd as u64;
            let n_layer = n_layer as u64;
            let n_ctx = n_ctx as u64;
            let n_vocab = n_vocab as u64;

            /// NOTE: The original code relies in promotion rules and automatic
            /// cast between int to float. What we do instead is use this macro
            /// to convert every term of the multiplication to f64, which should
            /// have enough precision bits to hold the final value, then cast to
            /// usize. I have observed a discrepancy between the ctx_size found
            /// using this code, and the one in llama.cpp. The number for rust
            /// ends up being slightly lower, but no "out of memory" errors are
            /// reported by ggml.
            macro_rules! mul {
                ($term:expr, $($terms:expr),*) => {
                    (($term as f64) $(* ($terms as f64))*) as u64
                };
            }

            fn ggml_type_sizef(x: ggml_raw::ggml_type) -> f64 {
                (unsafe { ggml_raw::ggml_type_sizef(x) }) as f64
            }

            let mut ctx_size: u64 = 0;

            ctx_size += mul!(n_embd, ggml_type_sizef(ggml::TYPE_F32)); // ln_f_g
            ctx_size += mul!(n_embd, ggml_type_sizef(ggml::TYPE_F32)); // ln_f_b

            ctx_size += mul!(n_embd, n_vocab, ggml_type_sizef(wtype)); // wte

            ctx_size += mul!(n_embd, n_vocab, ggml_type_sizef(wtype)); // Lmh_g
            ctx_size += mul!(n_vocab, ggml_type_sizef(ggml::TYPE_F32)); // Lmh_b

            ctx_size += mul!(n_layer, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // ln_1_g
            ctx_size += mul!(n_layer, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // ln_1_b

            ctx_size += mul!(n_layer, n_embd, n_embd, ggml_type_sizef(wtype)); // c_attn_q_proj_w
            ctx_size += mul!(n_layer, n_embd, n_embd, ggml_type_sizef(wtype)); // c_attn_k_proj_w
            ctx_size += mul!(n_layer, n_embd, n_embd, ggml_type_sizef(wtype)); // c_attn_v_proj_w

            ctx_size += mul!(n_layer, n_embd, n_embd, ggml_type_sizef(wtype)); // c_attn_proj_w

            ctx_size += mul!(n_layer, 4, n_embd, n_embd, ggml_type_sizef(wtype)); // c_mlp_fc_w
            ctx_size += mul!(n_layer, 4, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // c_mlp_fc_b

            ctx_size += mul!(n_layer, 4, n_embd, n_embd, ggml_type_sizef(wtype)); // c_mlp_proj_w_trans
            ctx_size += mul!(n_layer, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // c_mlp_proj_b

            ctx_size += mul!(n_ctx, n_layer, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // memory_k
            ctx_size += mul!(n_ctx, n_layer, n_embd, ggml_type_sizef(ggml::TYPE_F32)); // memory_v

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            load_progress_callback(LoadProgress::ContextSize {
                bytes: ctx_size.try_into()?,
            });

            ctx_size
        };

        // Initialize the context
        let context = ggml::Context::init(ctx_size as usize);

        let model = {
            let mut tensors = HashMap::new();

            // wte
            let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);

            let ln_f_g = context.new_tensor_1d(ggml::TYPE_F32, n_embd);
            let ln_f_b = context.new_tensor_1d(ggml::TYPE_F32, n_embd);

            let lmh_g = context.new_tensor_2d(wtype, n_embd, n_vocab);
            let lmh_b = context.new_tensor_1d(ggml::TYPE_F32, n_vocab);

            tensors.insert("transformer.wte.weight".to_owned(), tok_embeddings.share());
            tensors.insert("transformer.ln_f.weight".to_owned(), ln_f_g.share());
            tensors.insert("transformer.ln_f.bias".to_owned(), ln_f_b.share());
            tensors.insert("lm_head.weight".to_owned(), lmh_g.share());
            tensors.insert("lm_head.bias".to_owned(), lmh_b.share());

            let mut layers = Vec::new();
            for i in 0..n_layer {
                let layer = Layer {
                    ln_1_g: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
                    ln_1_b: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
                    c_attn_q_proj_w: context.new_tensor_2d(wtype, n_embd, n_embd),
                    c_attn_k_proj_w: context.new_tensor_2d(wtype, n_embd, n_embd),
                    c_attn_v_proj_w: context.new_tensor_2d(wtype, n_embd, n_embd),
                    c_attn_proj_w: context.new_tensor_2d(wtype, n_embd, n_embd),
                    c_mlp_fc_w:  context.new_tensor_2d(wtype, 4 * n_embd, n_embd),
                    c_mlp_fc_b: context.new_tensor_1d(ggml::TYPE_F32, 4 * n_embd),
                    c_mlp_proj_w_trans: context.new_tensor_2d(wtype, 4 * n_embd, n_embd),
                    c_mlp_proj_b: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
                };

                tensors.insert(format!("transformer.h.{i}.ln_1.weight"), layer.ln_1_g.share());
                tensors.insert(format!("transformer.h.{i}.ln_1.bias"), layer.ln_1_b.share());

                tensors.insert(format!("transformer.h.{i}.attn.q_proj.weight"), layer.c_attn_q_proj_w.share());
                tensors.insert(format!("transformer.h.{i}.attn.k_proj.weight"), layer.c_attn_k_proj_w.share());
                tensors.insert(format!("transformer.h.{i}.attn.v_proj.weight"), layer.c_attn_v_proj_w.share());

                tensors.insert(format!("transformer.h.{i}.attn.out_proj.weight"), layer.c_attn_proj_w.share());

                tensors.insert(format!("transformer.h.{i}.mlp.fc_in.weight"), layer.c_mlp_fc_w.share());
                tensors.insert(format!("transformer.h.{i}.mlp.fc_in.bias"), layer.c_mlp_fc_b.share());

                tensors.insert(format!("transformer.h.{i}.mlp.fc_out.weight"), layer.c_mlp_proj_w_trans.share());
                tensors.insert(format!("transformer.h.{i}.mlp.fc_out.bias"), layer.c_mlp_proj_b.share());

                layers.push(layer);
            }

            // key + value memory
            let n_mem = n_layer * n_ctx;
            let n_elements = n_embd * n_mem;

            let memory_k = context.new_tensor_1d(ggml::TYPE_F32, n_elements);
            let memory_v = context.new_tensor_1d(ggml::TYPE_F32, n_elements);

            let memory_size = memory_k.nbytes() + memory_v.nbytes();

            load_progress_callback(LoadProgress::MemorySize {
                bytes: memory_size,
                n_mem: n_mem.try_into()?,
            });

            Model {
                hparams,
                ln_f_g,
                ln_f_b,
                tok_embeddings,
                lmh_g,
                lmh_b,
                layers,
                memory_k,
                memory_v,
                tensors,
                _context: context,
            }
        };

        let mut total_size = 0;
        let mut n_tensors = 0;

        // Load weights
        loop {
            load_progress_callback(LoadProgress::TensorLoading {
                current_tensor: n_tensors.try_into()?,
                tensor_count: model.tensors.len(),
            });

            // NOTE: Implementation from #![feature(buf_read_has_data_left)]
            let is_eof = reader.fill_buf().map(|b| b.is_empty())?;

            if is_eof {
                break;
            }

            let n_dims = read_i32(&mut reader)?;
            let length = read_i32(&mut reader)?;
            let ftype = read_i32(&mut reader)?;

            let mut nelements = 1;
            let mut ne = [1i32, 1i32];
            for i in 0..n_dims {
                ne[i as usize] = read_i32(&mut reader)?;
                nelements *= ne[i as usize];
            }

            let tensor_name = read_string(&mut reader, length as usize)?;

            let Some(tensor) = model.tensors.get(&tensor_name)
                else {
                    return Err(LoadError::UnknownTensor { tensor_name, path: path.to_path_buf() });
                };

            if tensor.nelements() != nelements {
                return Err(LoadError::TensorWrongSize {
                    tensor_name,
                    path: path.to_path_buf()
                });
            }

            if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] != ne[1] {
                return Err(LoadError::TensorWrongSize {
                    tensor_name,
                    path: path.to_path_buf()
                });
            }

            fn ggml_type_size(t: ggml::Type) -> usize {
                unsafe { ggml_raw::ggml_type_size(t) }
            }

            fn ggml_blck_size(t: ggml::Type) -> i32 {
                unsafe { ggml_raw::ggml_blck_size(t) }
            }

            let bpe = match ftype {
                0 => ggml_type_size(ggml::TYPE_F32),
                1 => ggml_type_size(ggml::TYPE_F16),
                _ => {
                    return Err(LoadError::InvalidFtype {
                        ftype,
                        path: path.to_path_buf()
                    })
                }
            };

            if (nelements as usize * bpe) / ggml_blck_size(tensor.get_type()) as usize
                != tensor.nbytes()
            {
                return Err(LoadError::TensorWrongSize {
                    tensor_name,
                    path: path.to_path_buf()
                });
            }

            let data = tensor.data();

            // SAFETY: yolo, same as original code
            let slice = unsafe {
                std::slice::from_raw_parts_mut(data as *mut u8, tensor.nbytes())
            };
            reader.read_exact(slice)?;

            load_progress_callback(LoadProgress::TensorLoaded {
                current_tensor: n_tensors.try_into()?,
                tensor_count: model.tensors.len(),
            });
            total_size += tensor.nbytes();
            n_tensors += 1;
        }

        drop(reader);

        Ok((model, vocab))
    }

    pub fn inference_with_prompt(
        &self,
        vocab: &Vocabulary,
        params: &InferenceParameters,
        prompt: &str,
        rng: &mut impl rand::Rng,
        callback: impl Fn(OutputToken),
    ) {
        let embd_inp = self.tokenize(vocab, prompt);
        let mut logits = Vec::new();

        // determine the required inference memory per token:
        let mut mem_per_token = 0;
        let _ = self.evaluate(
            params.n_threads,
            0,
            &[0, 1, 2, 3],
            &mut logits,
            &mut mem_per_token,
        );

        let last_n_size = params.repeat_last_n;
        let mut last_n_tokens = vec![0 as TokenId; last_n_size];

        let mut remaining_tokens = usize::min(
            params.n_predict,
            self.hparams.n_ctx as usize - embd_inp.len(),
        );
        let mut input_consumed = 0;

        let mut n_past = 0;
        let mut embd = Vec::new();
        while remaining_tokens > 0 {
            // predict
            if embd.len() > 0 {
                self.evaluate(
                    params.n_threads,
                    n_past,
                    &embd,
                    &mut logits,
                    &mut mem_per_token,
                );
            }

            n_past += embd.len() as i32;
            embd.clear();

            if embd_inp.len() <= input_consumed {
                // out of input, sample next token
                let InferenceParameters {
                    top_k,
                    top_p,
                    repeat_penalty,
                    temp,
                    ..
                } = params;

                let n_vocab = self.hparams.n_vocab;

                let id = self.sample_top_p_top_k(
                    vocab,
                    &logits[logits.len() - n_vocab as usize..],
                    &last_n_tokens,
                    *repeat_penalty as f64,
                    *top_k,
                    *top_p as f64,
                    *temp as f64,
                    rng,
                );

                last_n_tokens.remove(0);
                last_n_tokens.push(id);

                // add it to the context
                embd.push(id);

                // decrement remaining sampling budget
                remaining_tokens -= 1;
            } else {
                // if here, it means we are still processing the input prompt
                while embd_inp.len() > input_consumed {
                    embd.push(embd_inp[input_consumed]);
                    last_n_tokens.remove(0);
                    last_n_tokens.push(embd_inp[input_consumed]);
                    input_consumed += 1;
                    if embd.len() > params.n_batch {
                        break;
                    }
                }
            }

            // display text
            let mut eot = false;
            for &id in &embd {
                let output_token = if id == 2 {
                    eot = true;
                    OutputToken::EndOfText
                } else {
                    OutputToken::Token(&vocab.mapping[id as usize])
                };
                callback(output_token);
            }

            if eot {
                break;
            }
        }
    }

    pub fn sample_top_p_top_k(
        &self,
        vocab: &Vocabulary,
        logits: &[f32],
        last_n_tokens: &[TokenId],
        repeat_penalty: f64,
        top_k: i32,
        top_p: f64,
        temp: f64,
        rng: &mut impl rand::Rng,
    ) -> TokenId {
        let n_logits = vocab.mapping.len();
        let mut logits_id = Vec::<(f64, TokenId)>::with_capacity(n_logits);

        {
            let scale = 1.0 / temp;
            for (i, &logit) in logits.iter().enumerate() {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                if last_n_tokens.contains(&(i as TokenId)) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logits_id.push((logit as f64 * scale * repeat_penalty, i as TokenId));
                    } else {
                        logits_id.push((logit as f64 * scale / repeat_penalty, i as TokenId));
                    }
                } else {
                    logits_id.push((logit as f64 * scale, i as TokenId));
                }
            }
        }

        // find the top K tokens
        {
            logits_id.partial_sort(top_k as usize, |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(top_k as usize);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f64::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f64> = logits_id
            .iter()
            .copied()
            .map(|(k, v)| (k - maxl).exp())
            .collect();
        let sum: f64 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }

    pub fn evaluate(
        &self,
        n_threads: i32,
        n_past: i32,
        embd_inp: &[TokenId],
        embd_w: &mut Vec<f32>,
        mem_per_token: &mut usize,
    ) {
        let N = embd_inp.len();

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_head,
            n_layer,
            n_rot,
            f16_: _,
        } = self.hparams;

        let mut buf_size = 256 * 1024 * 1024;
        if *mem_per_token > 0 && *mem_per_token * N > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * *mem_per_token as f64 * N as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::TYPE_I32, N as i32);
        unsafe { embd.write_data(bytemuck::cast_slice(embd_inp)) };

        let mut inpL = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        for il in 0..n_layer as usize {
            let mut cur: ggml::Tensor;

            // norm
            {
                cur = ctx0.op_norm(&inpL);

                // cur = ln_1_g*cur + ln_1_b
                cur = ctx0.op_add(
                    &ctx0.op_mul(&ctx0.op_repeat(&self.layers[il].ln_1_g, &cur), &cur),
                    &ctx0.op_repeat(&self.layers[il].ln_1_b, &cur)
                );
            }

            let inpSA = cur.share();

            // self-attention
            {
                let Qcur = ctx0.op_mul_mat(&ctx0.op_transpose(&self.layers[il].c_attn_q_proj_w), &cur);
                let Kcur = ctx0.op_mul_mat(&ctx0.op_transpose(&self.layers[il].c_attn_k_proj_w), &cur);
                let Vcur = ctx0.op_mul_mat(&ctx0.op_transpose(&self.layers[il].c_attn_v_proj_w), &cur);

                // store key and value to memory
                if N >= 1 {
                    let k = ctx0.op_view_1d(
                        &self.memory_k,
                        N as i32 * n_embd,
                        (self.memory_k.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    let v = ctx0.op_view_1d(
                        &self.memory_v,
                        N as i32 * n_embd,
                        (self.memory_v.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    gf.build_forward_expand(&ctx0.op_cpy(&Kcur, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&Vcur, &v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let Q = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_cpy(
                            &Qcur,
                            &ctx0.new_tensor_3d(ggml::TYPE_F32, n_embd / n_head, n_head, N as i32),
                        ),
                        n_past,
                        n_rot,
                        0,
                    ),
                    0, 2, 1, 3,
                );

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let K = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                &self.memory_k,
                                (n_past + N as i32) * n_embd,
                                il * n_ctx as usize
                                    * self.memory_k.element_size()
                                    * n_embd as usize,
                            ),
                            n_embd / n_head,
                            n_head,
                            n_past + N as i32,
                        ),
                        n_past,
                        n_rot,
                        1,
                    ),
                    0, 2, 1, 3,
                );

                // K * Q
                let KQ = ctx0.op_mul_mat(&K, &Q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let KQ_scaled = ctx0.op_scale(
                    &KQ,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                // KQ_masked = mask_past(KQ_scaled)
                let KQ_masked = ctx0.op_diag_mask_inf(&KQ_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let KQ_soft_max = ctx0.op_soft_max(&KQ_masked);

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                let V_trans = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &self.memory_v,
                            (n_past + N as i32) * n_embd,
                            il * n_ctx as usize * self.memory_v.element_size() * n_embd as usize,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + N as i32,
                    ),
                    1,
                    2,
                    0,
                    3,
                );

                // KQV = transpose(V) * KQ_soft_max
                let KQV = ctx0.op_mul_mat(&V_trans, &KQ_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let KQV_merged = ctx0.op_permute(&KQV, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                cur = ctx0.op_cpy(
                    &KQV_merged,
                    &ctx0.new_tensor_2d(ggml::TYPE_F32, n_embd, N as i32),
                );

                // projection (no bias)
                cur = ctx0.op_mul_mat(&ctx0.op_transpose(&self.layers[il].c_attn_proj_w), &cur);
            }

            //let inpFF = ctx0.op_add(&cur, &inpSA);
            let inpFF = cur.share();

            // feed-forward network
            {
                // note here we pass inpSA instead of cur
                cur = ctx0.op_mul_mat(&ctx0.op_transpose(&self.layers[il].c_mlp_fc_w), &inpSA);
                cur = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].c_mlp_fc_b, &cur), &cur);

                // GELU activation
                cur = ctx0.op_gelu(&cur);

                // projection
                // cur = proj_w*cur + proj_b
                cur = ctx0.op_mul_mat(&self.layers[il].c_mlp_proj_w_trans, &cur);
                cur = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].c_mlp_proj_b, &cur), &cur);
            }

            // self-attention + FF
            cur = ctx0.op_add(&cur, &inpFF);

            // input for next layer
            //inpL = cur;
            inpL = ctx0.op_add(&cur, &inpL);
        }

        // norm
        {
            inpL = ctx0.op_norm(&inpL);

            // inpL = ln_f_g*inpL + ln_f_b
            inpL = ctx0.op_add(
                &ctx0.op_mul(&ctx0.op_repeat(&self.ln_f_g, &inpL), &inpL),
                &ctx0.op_repeat(&self.ln_f_b, &inpL)
            )
        }

        // lm_head
        {
            inpL = ctx0.op_mul_mat(&self.lmh_g, &inpL);
            inpL = ctx0.op_add(&ctx0.op_repeat(&self.lmh_b, &inpL), &inpL);
        }

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&inpL);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        embd_w.resize(n_vocab as usize, 0.0);
        // SAFETY: yolo
        unsafe {
            inpL.read_data(
                n_vocab as usize * (N - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(embd_w),
            )
        };

        if *mem_per_token == 0 {
            *mem_per_token = ctx0.used_mem() / N;
        }
    }

    pub fn tokenize(&self, vocab: &Vocabulary, text: &str) -> Vec<TokenId> {
        // first split the text into words
        let re = Regex::new(r#"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+)"#).unwrap();
        let words: Vec<&str> = re.find_iter(text).map(|m| m.unwrap().as_str()).collect();

        // find the longest tokens that form the words:
        let mut tokens: Vec<TokenId> = Vec::new();
        for word in words.iter() {
            if word.len() == 0 {
                continue;
            }

            let mut i = 0;
            let n = word.len();
            while i < n {
                let mut j = n;
                while j > i {
                    let search: &str = &word[i..j];
                    let val = vocab.mapping.iter().enumerate().find(|(_tk_id, tk)| tk.as_str() == search);
                    if val.is_none() {
                        j -= 1;
                    } else {
                        tokens.push(val.unwrap().0 as TokenId);
                        i = j;
                        break;
                    }
                }

                if i == n {
                    break;
                }

                if j == i {
                    let sub = &word[i..(i+1)];
                    let val = vocab.mapping.iter().enumerate().find(|(_tk_id, tk)| tk.as_str() == sub);
                    if val.is_none() {
                        println!("Uh oh! Invalid token!");
                    } else {
                        tokens.push(val.unwrap().0 as TokenId);
                    }
                    i += 1;
                }
            }
        }

        return tokens;
    }
}