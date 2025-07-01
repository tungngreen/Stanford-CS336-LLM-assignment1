import torch
import time

from gpt2.transformer.attention import scaled_dot_product_attention

batch_size = 8
d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 512, 1024, 2048, 4096, 8192, 16384]

num_warmup_steps = 10
num_fw_passes = 100
num_bwd_passes = 100

def main(compile=False):
    results = {}
    
    attn_fn = torch.compile(scaled_dot_product_attention) if compile else scaled_dot_product_attention
    
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            try:
                q = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
                k = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
                v = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
                
                current_pass = 'fw'
                
                # Warmup
                with torch.no_grad():
                    for _ in range(num_warmup_steps):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()
                
                # Forward passes
                with torch.no_grad():
                    t0 = time.time()
                    
                    for _ in range(num_fw_passes):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()
                        
                    t1 = time.time()
                    
                avg_time = (t1 - t0) / num_fw_passes
                
                current_pass = 'bwd'
                
                for _ in range(num_warmup_steps):
                    out = attn_fn(q, k, v)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None
                    
                mem_total = 0.0
                torch.cuda.synchronize()
                
                t0 = time.time()
                for _ in range(num_bwd_passes):
                    out = attn_fn(q, k, v)
                    mem_total += torch.cuda.memory_allocated() / (1024 ** 2)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None
                
                t1 = time.time()
                avg_bwd_time = (t1 - t0) / num_bwd_passes
                avg_mem = mem_total / num_bwd_passes
                
                print(f"d_model={d_model}, seq_len={seq_len}, "
                      f"fw_time={avg_time:.4f}s, bwd_time={avg_bwd_time:.4f}s, "
                      f"mem={avg_mem:.2f}MB")
                
                results[(d_model, seq_len)] = {
                    "fw_time": avg_time,
                    "bwd_time": avg_bwd_time,
                    "mem": avg_mem
                }
                
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if current_pass == 'fw':
                        print(f"OOM for d_model={d_model}, seq_len={seq_len}")
                    else:
                        print(f"d_model={d_model}, seq_len={seq_len}, "
                              f"fw_time={avg_time:.4f}s")
                    continue
                else:
                    raise e
                
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention implementation.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the attention function.")
    
    args = parser.parse_args()
    
    main(compile=args.compile)
    
    # Save results to a file if needed
    # import json
    # with open("attention_benchmark_results.json", "w") as f:
    #     json.dump(results, f, indent=4)
                
            