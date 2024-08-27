use burn::{
    backend::{Autodiff, Wgpu},
    config::Config,
    data::dataset::Dataset,
    optim::AdamConfig,
    tensor::Tensor,
    train::LearnerBuilder,
};
// use guide::{
//     inference,
//     model::ModelConfig,
//     training::{self, TrainingConfig},
// };
#[derive(Debug, Config)]
struct ModelConfig {
    model: Tensor<Wgpu, 2>,

    #[config(default = 4)]
    epoch: usize,
}

fn main() {
    // let device = Default::default();
    let device = burn::backend::wgpu::WgpuDevice::default();
    let mut config = ModelConfig::new(Tensor::<Wgpu, 2>::from_data([[1., 2.], [1., 2.]], &device));
    let learner = LearnerBuilder::new("./sus")
        .devices(vec![device.clone()])
        .num_epochs(config.epoch)
    // Create a default Wgpu device
    // kk
    // println!("{a:?}")

    // All the training artifacts will be saved in this directory
    // let artifact_dir = "/tmp/guide";

    // Train the model
    // training::train::<MyAutodiffBackend>(
    // artifact_dir,
    // TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    // device.clone(),
    // );

    // // Infer the model
    // inference::infer::<MyBackend>(
    //     artifact_dir,
    //     device,
    //     burn::data::dataset::vision::MnistDataset::test()
    //         .get(42)
    //         .unwrap(),
    // );
}
