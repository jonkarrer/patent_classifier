use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Autodiff, Wgpu,
};

pub type MyDevice = Wgpu<AutoGraphicsApi, f32, i32>;
pub type MyBackend = Autodiff<MyDevice>;

pub fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}
