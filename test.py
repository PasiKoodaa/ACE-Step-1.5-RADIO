import os
import sys
import torch
import shutil
from acestep.handler import AceStepHandler


def main():
    print("Initializing AceStepHandler...")
    handler = AceStepHandler()
    
    # Find checkpoints
    checkpoints = handler.get_available_checkpoints()
    if checkpoints:
        project_root = checkpoints[0]
    else:
        # Fallback
        current_file = os.path.abspath(__file__)
        project_root = os.path.join(os.path.dirname(current_file), "checkpoints")
        
    print(f"Project root (checkpoints dir): {project_root}")
    
    # Find models
    models = handler.get_available_acestep_v15_models()
    if not models:
        print("No models found. Using default 'acestep-v15-turbo'.")
        model_name = "./acestep-v15-turbo"
    else:
        model_name = models[0]
        print(f"Found models: {models}")
        print(f"Using model: {model_name}")
        
    # Initialize service
    device = "xpu"
    print(f"Using device: {device}")
    
    status, enabled = handler.initialize_service(
        project_root=project_root,
        config_path=model_name,
        device=device,
        init_llm=True,
        use_flash_attention=False, # Default in UI
        compile_model=True
    )
    
    if not enabled:
        print(f"Error initializing service: {status}")
        return
    
    print(status)
    print("Service initialized successfully.")
    
    # Prepare inputs
    captions = "A soft pop arrangement led by light, fingerpicked guitar sets a gentle foundation, Airy keys subtly fill the background, while delicate percussion adds warmth, The sweet female voice floats above, blending naturally with minimal harmonies in the chorus for an intimate, uplifting sound"
    
    lyrics = """[Intro]

[Verse 1]
风吹动那年仲夏
翻开谁青涩喧哗
白枫书架
第七页码

[Verse 2]
珍藏谁的长发
星夜似手中花洒
淋湿旧忆木篱笆
木槿花下
天蓝发夹
她默认了他

[Bridge]
时光将青春的薄荷红蜡
匆匆地融化
她却沉入人海再无应答
隐没在天涯

[Chorus]
燕子在窗前飞掠
寻不到的花被季节带回
拧不干的思念如月
初恋颜色才能够描绘

木槿在窗外落雪
倾泻道别的滋味
闭上眼听见微咸的泪水
到后来才知那故梦珍贵

[Outro]"""

    seeds = "320145306, 1514681811"
    
    print("Starting generation...")
    
    # Call generate_music
    results = handler.generate_music(
        captions=captions,
        lyrics=lyrics,
        bpm=90,
        key_scale="A major",
        time_signature="4",
        vocal_language="zh",
        inference_steps=8,
        guidance_scale=7.0,
        use_random_seed=False,
        seed=seeds,
        audio_duration=120,
        batch_size=1,
        task_type="text2music",
        cfg_interval_start=0.0,
        cfg_interval_end=0.95,
        audio_format="wav"
    )
    
    # Unpack results
    (audio1, audio2, saved_files, info, status_msg, seed_val, 
     align_score1, align_text1, align_plot1, 
     align_score2, align_text2, align_plot2) = results
     
    print("\nGeneration Complete!")
    print(f"Status: {status_msg}")
    print(f"Info: {info}")
    print(f"Seeds used: {seed_val}")
    print(f"Saved files: {saved_files}")
    
    # Copy files
    for f in saved_files:
        if os.path.exists(f):
            dst = os.path.basename(f)
            shutil.copy(f, dst)
            print(f"Saved output to: {os.path.abspath(dst)}")

if __name__ == "__main__":
    main()
