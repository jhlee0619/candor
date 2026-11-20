import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


# ===================== WAN 2.1 Video Config =====================


def general_ocr_sd3_8gpu():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 7.0

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 4
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number!"
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3_self_8gpu'
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_8gpu_d_paper():
    """CANDOR: reward D = log MSE(empty prompt) - log MSE(prompt); rollout CFG 4.5, last ceil(0.6 * 9) = 6 steps, w(t) = 1."""
    config = general_ocr_sd3_8gpu()
    config.sample.guidance_scale = 4.5
    config.train.suffix_ratio = 0.6
    config.train.sds = {"k": 8, "use_step_stride": 1, "scale": 1.0, "g_weight": 0.0, "d_weight": 1.0, "t_weight": "uniform"}
    config.save_dir = 'logs/ocr/sd3_d_paper_8gpu'
    config.activation_checkpointing = True  # memory only
    config.stop_global_step = 2000
    return config


def get_config(name):
    return globals()[name]()
