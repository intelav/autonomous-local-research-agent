"""
Central registry for all research domains.

Add a new domain here and it automatically propagates to:
- query classification
- arXiv search topics
- digest/tag generation
"""

DOMAINS = {
    "gpu_optimization": {
        "description": "CUDA programming, GPU kernels, compilers, profiling",
        "keywords": [
            "cuda", "gpu", "triton", "kernel", "nsight",
            "tensor core", "ptx", "nvcc", "compiler", "profiling", "tunix"
        ],
        "arxiv_topics": [
            "AI compiler optimization Triton kernel",
            "GPU performance profiling Nsight Compute",
            "NVIDIA GPU architecture Hopper Blackwell Tensor Core",
            "science AI for HPC and simulation",
        ],
    },

    "foundation_models": {
        "description": "Vision and multimodal foundation models",
        "keywords": [
            "transformer", "vit", "dinov2", "sam", "sam2",
            "foundation model", "multimodal", "llama", "whisper"
        ],
        "arxiv_topics": [
            "DINOv2 SAM2 multimodal foundation models",
            "semantic segmentation transformer",
            "foundation models for vision and language",
        ],
    },

    "geospatial_ai": {
        "description": "Satellite imagery and remote sensing AI",
        "keywords": [
            "satellite", "geospatial", "remote sensing",
            "change detection", "visual search", "earth observation"
        ],
        "arxiv_topics": [
            "satellite image change detection",
            "visual search in satellite imagery",
        ],
    },

    "vision_models": {
        "description": "Object detection, segmentation, YOLO, CNNs",
        "keywords": [
            "yolo", "object detection", "segmentation",
            "cnn", "mask", "panoptic"
        ],
        "arxiv_topics": [
            "YOLO object detection evolution",
            "object detection and segmentation models",
        ],
    },

    "ml_algorithms": {
        "description": "Classical machine learning algorithms",
        "keywords": [
            "svm", "naive bayes", "random forest",
            "xgboost", "classification", "regression"
        ],
        "arxiv_topics": [
            "machine learning algorithms SVM random forest XGBoost",
        ],
    },

    "deep_learning": {
        "description": "Neural networks and learning paradigms",
        "keywords": [
            "deep learning", "cnn", "rnn", "lstm",
            "gan", "reinforcement learning"
        ],
        "arxiv_topics": [
            "deep learning neural networks GAN reinforcement learning",
        ],
    },

    "ai_frameworks": {
        "description": "AI frameworks and runtimes",
        "keywords": [
            "pytorch", "jax", "tensorflow",
            "pallas", "numpy", "pandas"
        ],
        "arxiv_topics": [
            "AI Frameworks like Pytorch Tensorflow",
            "TPU JAX Pallas Kernels",
        ],
    },

    "agentic_ai": {
        "description": "Agentic systems and orchestration frameworks",
        "keywords": [
            "agentic", "autonomous", "mcp",
            "langchain", "langgraph", "llamaindex"
        ],
        "arxiv_topics": [
            "AI Agentic systems like MCP Langgraph Langchain LlamaIndex",
            "agentic GPU optimization AI workloads",
        ],
    },

    "security_crypto": {
        "description": "Cryptography, HSMs, and AI security",
        "keywords": [
            "hsm", "cryptography", "encryption",
            "key management", "pkcs", "fips", "security"
        ],
        "arxiv_topics": [
            "HSM Cryptography and AI Security",
        ],
    },
}

DEFAULT_DOMAIN = "general"
