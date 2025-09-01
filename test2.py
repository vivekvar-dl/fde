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
    
    # Method 1: Use balanced memory allocation
    print("\n=== Loading with balanced memory allocation ===")
    
    # Get available memory for each GPU
    max_memory = get_balanced_memory(
        model_id,
        max_memory=None,
        no_split_module_classes=["BloomBlock", "LlamaDecoderLayer", "GPTNeoXLayer"],  # Common layer types
        dtype=torch.float16,
        low_zero=(False, False),  # Don't use CPU/disk offload
    )
    
    print(f"Balanced memory allocation: {max_memory}")
    
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
    """Alternative method if first one doesn't work"""
    print("\n=== Alternative Loading Method ===")
    
    # Force 50/50 split by specifying device map
    device_map = "balanced_low_0"  # This forces more balanced distribution
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        offload_folder="./offload",  # Use disk offload if needed
        offload_state_dict=True
    )
    
    return model

def create_pipeline_with_model(model, model_id):
    """Create pipeline with pre-loaded model"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Create pipeline with existing model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
    
    return pipe

# Main execution
if __name__ == "__main__":
    model_id = "bigscience/bloom-176b"  # Change to your desired model
    
    try:
        # Try main method first
        model = setup_multi_gpu_model(model_id)
        check_gpu_distribution(model)
        
        # Test if both GPUs are being used
        gpu_0_usage = torch.cuda.memory_allocated(0) / 1024**3
        gpu_1_usage = torch.cuda.memory_allocated(1) / 1024**3
        
        if gpu_1_usage < 1.0:  # Less than 1GB on GPU 1
            print("\n⚠️  GPU 1 not being used enough, trying alternative method...")
            del model
            torch.cuda.empty_cache()
            model = alternative_loading_method(model_id)
            check_gpu_distribution(model)
        
        # Create pipeline and test
        pipe = create_pipeline_with_model(model, model_id)
        
        print("\n=== Testing Generation ===")
        prompt = "Explain quantum mechanics clearly and concisely."
        outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        print(f"Generated: {outputs[0]['generated_text'][len(prompt):]}")
        
        # Final GPU usage check
        print(f"\nFinal GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB allocated")
        
        # Check if both GPUs are being used
        both_gpus_used = True
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            if allocated < 1.0:
                both_gpus_used = False
                break
        
        if both_gpus_used:
            print("✅ Both GPUs are being utilized!")
        else:
            print("❌ Both GPUs are not being utilized properly")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying with quantization...")
        
        # Fallback with quantization
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
        
        check_gpu_distribution(model)
