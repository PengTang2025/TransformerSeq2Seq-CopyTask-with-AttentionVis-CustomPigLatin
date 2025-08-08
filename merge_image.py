import os
from PIL import Image

def merge_images(image_name, image_dir, output_path):
    images = []
    for i in range(8):
        # 读取每个头的注意力热图
        # read each head's attention heatmap
        # word_encoder_attn_head_i.png, word_decoder_self-attn_head_i.png, word_decoder_cross-attn_head_i.png
        encoder_image = Image.open(os.path.join(image_dir, f"{word}_encoder_attn_head_{i+1}.png"))
        decoder_self_image = Image.open(os.path.join(image_dir, f"{word}_decoder_self-attn_head_{i+1}.png"))
        decoder_cross_image = Image.open(os.path.join(image_dir, f"{word}_decoder_cross-attn_head_{i+1}.png"))
        
        # 将三张图像合并为一张图像
        # combine the three images into one
        combined_image = Image.new('RGB', (encoder_image.width + decoder_self_image.width + decoder_cross_image.width, encoder_image.height))
        combined_image.paste(encoder_image, (0, 0)) 
        combined_image.paste(decoder_self_image, (encoder_image.width, 0))
        combined_image.paste(decoder_cross_image, (encoder_image.width + decoder_self_image.width, 0))

        
        images.append(combined_image)

    # 将所有行合并为一张图像
    # merge all rows into one image
    total_width = images[0].width
    total_height = images[0].height * len(images)
    final_image = Image.new('RGB', (total_width, total_height))

    for i, img in enumerate(images):
        final_image.paste(img, (0, i * img.height))

    final_image.save(f"{output_path}/{word}_merged_attention_heatmaps.png")
    
if __name__ == "__main__":
    image_dir = "test_visualization"  # 图像所在目录 the directory where images are stored
    output_path = "merged_images"  # 输出目录 the output directory for merged images
    os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在 make sure the output directory exists
    # word_encoder_attn_head_i.png, word_decoder_self-attn_head_i.png, word_decoder_cross-attn_head_i.png
    # 先读文件名，获取word列表
    # read the filenames first to get the list of words
    words = set()
    for filename in os.listdir(image_dir):
        word = filename.split('_')[0]  # 分割文件名 to get the word
        if word not in words:
            words.add(word)
    for word in words:
        merge_images(word, image_dir, output_path)
        print(f"Merged image {word} saved to {output_path}")