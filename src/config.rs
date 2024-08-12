use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Autodiff, Wgpu,
};
use tokenizers::Tokenizer;

pub type MyDevice = Wgpu<AutoGraphicsApi, f32, i32>;
pub type MyBackend = Autodiff<MyDevice>;

pub fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}

pub fn init_tokenizer() -> Tokenizer {
    Tokenizer::from_pretrained("bert-base-cased", None).expect("Tokenizer not initialized")
}

pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub device: WgpuDevice,
}
