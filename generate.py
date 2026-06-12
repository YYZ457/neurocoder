"""
NeuroCoder Interactive Generator
================================
Load a trained NeuroCoder model and generate Python code interactively.

Usage:
    python generate.py --checkpoint D:/NeuroCoder/checkpoints/final.pt
    python generate.py --checkpoint latest.pt --interactive
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import NeuroCoderConfig, CONFIG_4060
from model import NeuroCoder
from data import CodeTokenizer


class CodeGenerator:
    """Interactive Python code generator."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Get config
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = CONFIG_4060

        # Create model
        print("Building model...")
        self.model = NeuroCoder(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = CodeTokenizer()

        self.config = config
        print(f"[OK] Model ready ({self.model.get_num_params()/1e6:.1f}M params)")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        """Generate Python code from a text prompt."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Generate
        output = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        output_ids = output[0].tolist()
        full_text = self.tokenizer.decode(output_ids, skip_special=True)

        # Extract only the newly generated part
        generated = full_text[len(prompt):] if full_text.startswith(prompt) else full_text

        return generated

    def interactive_mode(self):
        """Interactive REPL for code generation."""
        print("\n" + "=" * 60)
        print("  NeuroCoder Interactive Mode")
        print("  Type your prompt (or 'quit' to exit)")
        print("  Type '!temp 0.5' to change temperature")
        print("  Type '!help' for all commands")
        print("=" * 60)

        temperature = 0.7
        top_p = 0.95
        top_k = 50
        max_tokens = 256

        while True:
            try:
                user_input = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("!"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "!quit" or cmd == "!exit":
                    print("  Goodbye!")
                    break
                elif cmd == "!help":
                    print("""
  Commands:
    !quit, !exit    - Exit
    !temp <float>    - Set temperature (0.1-2.0, default 0.7)
    !top_p <float>   - Set nucleus sampling (0.0-1.0, default 0.95)
    !top_k <int>     - Set top-k sampling (0-200, default 50)
    !max_tokens <int> - Set max new tokens (16-1024, default 256)
    !show            - Show current settings
    !reset           - Reset to defaults
                    """)
                elif cmd == "!temp" and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"  Temperature = {temperature}")
                elif cmd == "!top_p" and len(parts) > 1:
                    top_p = float(parts[1])
                    print(f"  Top-p = {top_p}")
                elif cmd == "!top_k" and len(parts) > 1:
                    top_k = int(parts[1])
                    print(f"  Top-k = {top_k}")
                elif cmd == "!max_tokens" and len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"  Max tokens = {max_tokens}")
                elif cmd == "!show":
                    print(f"""
  Current settings:
    Temperature:  {temperature}
    Top-p:        {top_p}
    Top-k:        {top_k}
    Max tokens:   {max_tokens}
                    """)
                elif cmd == "!reset":
                    temperature = 0.7
                    top_p = 0.95
                    top_k = 50
                    max_tokens = 256
                    print("  Settings reset to defaults")
                continue

            # Generate
            print(f"\n  Generating (temp={temperature}, top_p={top_p}, max_tokens={max_tokens})...")
            print("  " + "-" * 56)

            generated = self.generate(
                prompt=user_input,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Display
            print(generated)
            print("  " + "-" * 56)


def main():
    parser = argparse.ArgumentParser(description="NeuroCoder Code Generator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="",
                        help="Generate code for this prompt (non-interactive)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive REPL")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    generator = CodeGenerator(args.checkpoint, device=args.device)

    if args.interactive or not args.prompt:
        generator.interactive_mode()
    else:
        print(f"\n  Prompt: {args.prompt}")
        print("  " + "-" * 56)
        generated = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(generated)
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
