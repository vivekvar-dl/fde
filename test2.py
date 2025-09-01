from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_multi_gpu_model(model_id):
    """Load model distributed across both GPUs"""
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Method 1: Simple auto device mapping
    print("\n=== Loading with auto device mapping ===")
    
    # Load model with auto device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    return model

def setup_multi_gpu_model_with_balanced_memory(model_id):
    """Alternative method using balanced memory allocation"""
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print("\n=== Loading with balanced memory allocation ===")
    
    # First, load the model config to get the architecture info
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Create a temporary model to get memory requirements
    # Note: This approach loads the model twice, but ensures proper memory calculation
    try:
        # Get available memory for each GPU (specify actual memory limits)
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            # Reserve some memory for other operations
            available_memory = torch.cuda.get_device_properties(i).total_memory * 0.85  # Use 85% of total memory
            max_memory[i] = int(available_memory)
        
        print(f"Memory allocation: {max_memory}")
        
        # Load model with explicit memory mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        return model
        
    except Exception as e:
        print(f"Balanced memory allocation failed: {e}")
        print("Falling back to simple auto device mapping...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        return model

def check_gpu_distribution(model):
    """Check how model is distributed across GPUs"""
    print("\n=== Model Distribution Check ===")
    
    device_usage = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        device = str(param.device)
        param_count = param.numel()
        total_params += param_count
        
        if device not in device_usage:
            device_usage[device] = {'params': 0, 'layers': []}
        
        device_usage[device]['params'] += param_count
        device_usage[device]['layers'].append(name)
    
    print(f"Total parameters: {total_params:,}")
    print("\nParameters per device:")
    for device, info in device_usage.items():
        percentage = (info['params'] / total_params) * 100
        print(f"{device}: {info['params']:,} parameters ({percentage:.1f}%)")
        print(f"  Sample layers: {info['layers'][:3]}...")
    
    # Check GPU memory usage
    print(f"\nCurrent GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

def alternative_loading_method(model_id):
    """Alternative method with manual device mapping"""
    print("\n=== Alternative Loading Method ===")
    
    # Manual device mapping for more control
    # This example assumes a transformer model structure
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 1,
        "model.layers.9": 1,
        "model.layers.10": 1,
        "model.layers.11": 1,
        "model.layers.12": 1,
        "model.layers.13": 1,
        "model.layers.14": 1,
        "model.layers.15": 1,
        "model.norm": 1,
        "lm_head": 1,
    }
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model
    except Exception as e:
        print(f"Manual device mapping failed: {e}")
        # Fall back to auto mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        return model

def create_pipeline_with_model(model, model_id):
    """Create pipeline with pre-loaded model"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create pipeline with existing model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
    
    return pipe

def install_bitsandbytes():
    """Install bitsandbytes if needed"""
    try:
        import bitsandbytes
        print("bitsandbytes is already installed")
        return True
    except ImportError:
        print("Installing bitsandbytes...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
            print("bitsandbytes installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install bitsandbytes: {e}")
            return False

def setup_quantized_model(model_id):
    """Setup model with quantization"""
    if not install_bitsandbytes():
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model
    except Exception as e:
        print(f"Quantization setup failed: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Use a smaller model for testing - BLOOM-176B is extremely large (350GB+)
    # model_id = "bigscience/bloom-176b"  # This requires ~350GB of memory
    model_id = "openai/gpt-oss-120b"  # Much smaller for testing
    # model_id = "microsoft/DialoGPT-medium"  # Even smaller alternative
    
    print(f"Loading model: {model_id}")
    
    try:
        # Try main method first
        model = setup_multi_gpu_model(model_id)
        check_gpu_distribution(model)
        
        # Test if both GPUs are being used
        gpu_usage = []
        for i in range(torch.cuda.device_count()):
            usage = torch.cuda.memory_allocated(i) / 1024**3
            gpu_usage.append(usage)
        
        min_usage_threshold = 0.5  # At least 500MB
        if any(usage < min_usage_threshold for usage in gpu_usage):
            print(f"\n⚠️  Some GPUs not being used enough (usage: {gpu_usage}), trying alternative method...")
            del model
            torch.cuda.empty_cache()
            model = alternative_loading_method(model_id)
            check_gpu_distribution(model)
        
        # Create pipeline and test
        pipe = create_pipeline_with_model(model, model_id)
        
        print("\n=== Testing Generation ===")
        prompt = "Explain quantum mechanics in simple terms:"
        outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, pad_token_id=pipe.tokenizer.eos_token_id)
        print(f"Generated: {outputs[0]['generated_text']}")
        
        # Final GPU usage check
        print(f"\nFinal GPU Memory Usage:")
        all_gpus_used = True
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB allocated")
            if allocated < min_usage_threshold:
                all_gpus_used = False
        
        if all_gpus_used:
            print("✅ All available GPUs are being utilized!")
        else:
            print("❌ Not all GPUs are being utilized optimally")
            
    except Exception as e:
        print(f"Error with main approach: {e}")
        print("\nTrying with quantization...")
        
        model = setup_quantized_model(model_id)
        if model is not None:
            check_gpu_distribution(model)
            pipe = create_pipeline_with_model(model, model_id)
            
            # Test generation
            prompt = "Explain quantum mechanics in simple terms:"
            outputs = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7, pad_token_id=pipe.tokenizer.eos_token_id)
            print(f"Generated: {outputs[0]['generated_text']}")
        else:
            print("❌ All methods failed. Consider using a smaller model or installing missing dependencies.")
