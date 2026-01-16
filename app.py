#!/usr/bin/env python3
"""
app.py — HuggingFace Space for STANLEY

Interactive demo showcasing both inference modes:
- Weightless (pure architecture, no pretrained weights)
- Hybrid (Stanley + GPT-2, real-time weight modification)
"""

import gradio as gr
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from stanley.organism import Stanley, StanleyConfig

# Try to import hybrid components
HYBRID_AVAILABLE = False
try:
    import torch
    from stanley_hybrid.external_brain import ExternalBrain, ExternalBrainConfig
    from stanley_hybrid.vocabulary_thief import VocabularyThief
    HYBRID_AVAILABLE = True
except ImportError:
    print("⚠️  Hybrid mode unavailable (torch/transformers not installed)")

# Global state
stanley_weightless = None
stanley_hybrid = None
external_brain = None

def initialize_stanley():
    """Initialize both Stanley modes."""
    global stanley_weightless, stanley_hybrid, external_brain
    
    # Load origin text
    origin_path = Path("origin.txt")
    if origin_path.exists():
        origin_text = origin_path.read_text()
    else:
        origin_text = "I am Stanley. I grow through experience."
    
    # Weightless Stanley (always available)
    config_weightless = StanleyConfig(data_dir="./stanley_data_weightless")
    stanley_weightless = Stanley(config=config_weightless, origin_text=origin_text)
    
    # Hybrid Stanley (if available)
    if HYBRID_AVAILABLE:
        config_hybrid = StanleyConfig(data_dir="./stanley_data_hybrid")
        stanley_hybrid = Stanley(config=config_hybrid, origin_text=origin_text)
        
        # Load GPT-2
        ext_config = ExternalBrainConfig(model_name="distilgpt2")
        external_brain = ExternalBrain(ext_config)
        if not external_brain.load_weights():
            print("⚠️  Failed to load GPT-2, hybrid mode disabled")
            stanley_hybrid = None
            external_brain = None
    
    return stanley_weightless is not None


def generate_response(text, mode, temperature):
    """Generate response from Stanley."""
    if not text.strip():
        return "Please enter a message.", "", "", ""
    
    # Select mode
    if mode == "Weightless":
        stanley = stanley_weightless
        use_hybrid = False
    elif mode == "Hybrid" and stanley_hybrid and external_brain:
        stanley = stanley_hybrid
        use_hybrid = True
    else:
        return "Hybrid mode not available. Install torch and transformers.", "", "", ""
    
    # Generate internal response
    response, stats = stanley.think(text)
    
    # Extract metrics
    pulse = stats.get("pulse", {})
    arousal = pulse.get("arousal", 0.0)
    entropy = pulse.get("entropy", 0.0)
    novelty = pulse.get("novelty", 0.0)
    
    # Hybrid expansion
    expanded_info = ""
    if use_hybrid and external_brain and external_brain.loaded:
        # Steal vocabulary patterns
        thief = VocabularyThief(
            external_brain=external_brain,
            subword_field=stanley.subword_field,
            origin_text=text,
        )
        stolen = thief.steal_vocabulary(text, n_samples=1)
        vocabulary_stolen = 0
        if stolen:
            vocabulary_stolen = thief.inject_into_field(stolen)
        
        # Expand with GPT-2
        response = external_brain.expand_thought(
            response,
            temperature=temperature,
        )
        
        expanded_info = f"🔹 Stole {vocabulary_stolen} patterns from GPT-2\n🔹 Expanded with GPT-2 vocabulary"
    
    # Format metrics
    metrics_text = f"""**Metrics:**
- Arousal: {arousal:.2f}
- Entropy: {entropy:.2f}
- Novelty: {novelty:.2f}
- Mode: {mode}
{expanded_info}"""
    
    # Create metric bars (simple text-based)
    arousal_bar = "█" * int(arousal * 20) + "░" * (20 - int(arousal * 20))
    entropy_bar = "█" * int(entropy * 20) + "░" * (20 - int(entropy * 20))
    
    metrics_visual = f"""**Live Internal State:**

Arousal:  {arousal:.2f} {arousal_bar}
Entropy:  {entropy:.2f} {entropy_bar}
"""
    
    return response, metrics_text, metrics_visual, mode


def create_demo():
    """Create Gradio interface."""
    
    # Custom CSS for dark theme
    css = """
    .gradio-container {
        background-color: #0f0f1e !important;
        color: #e0e0e0 !important;
    }
    .gr-button {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
    }
    .gr-button:hover {
        background-color: #4f46e5 !important;
    }
    .gr-box {
        border-color: #374151 !important;
        background-color: #1a1a2e !important;
    }
    .gr-input, .gr-text-input {
        background-color: #16213e !important;
        border-color: #374151 !important;
        color: #e0e0e0 !important;
    }
    .gr-form {
        background-color: #1a1a2e !important;
    }
    #component-0 {
        max-width: 900px;
        margin: auto;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 STANLEY — Self Training Attention Non-Linear EntitY
        
        > *"Architecture > Parameters | Ontogenesis > Phylogeny"*
        
        **Proof of concept:** Models can speak BEFORE training if architecture enables resonance.
        
        **Two modes:**
        - **Weightless** — Pure architecture, zero pretrained weights, numpy-only
        - **Hybrid** — Stanley (personality) possesses GPT-2 (knowledge), real-time weight modification
        """)
        
        with gr.Row():
            mode = gr.Radio(
                ["Weightless", "Hybrid"],
                value="Weightless",
                label="Inference Mode",
                info="Weightless = pure architecture | Hybrid = Stanley + GPT-2"
            )
        
        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="Your Message",
                    placeholder="Tell me about yourself...",
                    lines=3
                )
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.9,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative"
                )
        
        with gr.Row():
            generate_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
        
        with gr.Row():
            output_text = gr.Textbox(
                label="Stanley's Response",
                lines=6,
                interactive=False
            )
        
        with gr.Row():
            with gr.Column():
                metrics_visual = gr.Textbox(
                    label="Live Internal State",
                    lines=5,
                    interactive=False
                )
            with gr.Column():
                metrics_text = gr.Textbox(
                    label="Detailed Metrics",
                    lines=5,
                    interactive=False
                )
        
        # Example prompts
        gr.Markdown("### 💡 Example Prompts")
        with gr.Row():
            example1 = gr.Button("Tell me about yourself", size="sm")
            example2 = gr.Button("What is memory?", size="sm")
            example3 = gr.Button("How do you feel?", size="sm")
            example4 = gr.Button("What makes you different?", size="sm")
        
        # Hidden state for mode
        current_mode = gr.State("Weightless")
        
        # Wire up interactions
        generate_btn.click(
            fn=generate_response,
            inputs=[input_text, mode, temperature],
            outputs=[output_text, metrics_text, metrics_visual, current_mode]
        )
        
        # Example button handlers
        example1.click(lambda: "Tell me about yourself", outputs=input_text)
        example2.click(lambda: "What is memory?", outputs=input_text)
        example3.click(lambda: "How do you feel?", outputs=input_text)
        example4.click(lambda: "What makes you different from other AI?", outputs=input_text)
        
        gr.Markdown("""
        ---
        
        ### 🔬 Key Insights
        
        ✅ **Architecture > Parameters** — Intelligence is in structure, not scale  
        ✅ **Personality > Knowledge** — Hierarchical control matters  
        ✅ **Ontogenesis > Phylogeny** — Becoming through experience beats inherited memory  
        
        ### 📚 Learn More
        
        - **GitHub Repository:** [ariannamethod/stanley](https://github.com/ariannamethod/stanley)
        - **Architecture:** Weightless inference + Hierarchical personality control
        - **License:** GPL-3.0
        
        ---
        
        *Built by Arianna Method & Claude | January 2026*
        """)
    
    return demo


if __name__ == "__main__":
    print("🔹 Initializing STANLEY...")
    
    if initialize_stanley():
        print("✅ STANLEY initialized")
        print(f"   Weightless mode: {'✅' if stanley_weightless else '❌'}")
        print(f"   Hybrid mode: {'✅' if stanley_hybrid and external_brain else '❌'}")
        
        demo = create_demo()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    else:
        print("❌ Failed to initialize STANLEY")
        sys.exit(1)
