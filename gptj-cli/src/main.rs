use std::io::Write;
use rand::rngs::StdRng;

use cli_args::CLI_ARGS;
use gptj_rs::InferenceParameters;
use rand::{SeedableRng};

mod cli_args;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = &*CLI_ARGS;

    let inference_params = InferenceParameters {
        n_threads: args.num_threads as i32,
        n_predict: args.num_predict,
        n_batch: args.batch_size,
        top_k: args.top_k as i32,
        top_p: args.top_p,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
    };

    let prompt = if let Some(path) = &args.prompt_file {
        match std::fs::read_to_string(path) {
            Ok(prompt) => prompt,
            Err(err) => {
                eprintln!("Could not read prompt file at {path}. Error {err}");
                std::process::exit(1);
            }
        }
    } else if let Some(prompt) = &args.prompt {
        prompt.clone()
    } else {
        eprintln!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    };

    let (model, vocab) =
        gptj_rs::Model::load(&args.model_path, |progress| {
            use gptj_rs::LoadProgress;
            match progress {
                LoadProgress::HyperParamsLoaded(hparams) => {
                    log::info!("Loaded HyperParams {hparams:#?}")
                }
                LoadProgress::BadToken { index } => {
                    log::info!("Warning: Bad token in vocab at index {index}")
                }
                LoadProgress::ContextSize { bytes } => log::info!(
                    "ggml ctx size = {:.2} MB\n",
                    bytes as f64 / (1024.0 * 1024.0)
                ),
                LoadProgress::MemorySize { bytes, n_mem } => log::info!(
                    "Memory size: {} MB {}",
                    bytes as f32 / 1024.0 / 1024.0,
                    n_mem
                ),
                LoadProgress::TensorLoading {
                    current_tensor,
                    tensor_count,
                } => {

                },
                LoadProgress::TensorLoaded {
                    current_tensor,
                    tensor_count,
                    ..
                } => {
                    if current_tensor % 8 == 0 {
                        log::info!("Loaded tensor {current_tensor}/{tensor_count}");
                    }
                }
            }
        })
            .expect("Could not load model");

    log::info!("Model fully loaded!");

    let mut rng = StdRng::seed_from_u64(args.seed);
    model.inference_with_prompt(&vocab, &inference_params, &prompt, &mut rng, |t| {
        print!("{t}");
        std::io::stdout().flush().unwrap();
    });
    println!();
}