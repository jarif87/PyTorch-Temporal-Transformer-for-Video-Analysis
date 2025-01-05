import gradio as gr
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import VideoMAEForVideoClassification
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "model"
loaded_model = VideoMAEForVideoClassification.from_pretrained(model_path)
loaded_model = loaded_model.to(device)
loaded_model.eval()

label_names = [
   'Archery', 'BalanceBeam', 'BenchPress', 'ApplyEyeMakeup', 'BasketballDunk',
   'BandMarching', 'BabyCrawling', 'ApplyLipstick', 'BaseballPitch', 'Basketball'
]

def load_video(video_path):
   try:
       if not os.path.exists(video_path):
           raise ValueError(f"Video file not found: {video_path}")
           
       video = EncodedVideo.from_path(video_path)
       video_data = video.get_clip(start_sec=0, end_sec=video.duration)
       return video_data['video']
   except Exception as e:
       raise ValueError(f"Error loading video: {str(e)}")

def preprocess_video(video_frames):
   try:
       transform_temporal = UniformTemporalSubsample(16)
       video_frames = transform_temporal(video_frames)
       video_frames = video_frames.float() / 255.0
       if video_frames.shape[0] == 3:
           video_frames = video_frames.permute(1, 0, 2, 3)
       mean = torch.tensor([0.485, 0.456, 0.406])
       std = torch.tensor([0.229, 0.224, 0.225])
       for t in range(video_frames.shape[0]):
           video_frames[t] = F_t.normalize(video_frames[t], mean, std)
       video_frames = torch.stack([
           F_t.resize(frame, [224, 224], antialias=True) 
           for frame in video_frames
       ])
       video_frames = video_frames.unsqueeze(0)
       return video_frames
   except Exception as e:
       raise ValueError(f"Error preprocessing video: {str(e)}")

def predict_video(video):
   if video is None:
       return "Please upload a video file."
       
   try:
       video_data = load_video(video)
       processed_video = preprocess_video(video_data)
       processed_video = processed_video.to(device)
       with torch.no_grad():
           outputs = loaded_model(processed_video)
           logits = outputs.logits
           probabilities = F.softmax(logits, dim=-1)[0]
           top_3 = torch.topk(probabilities, 3)
       results = [
           f"{label_names[idx.item()]}: {prob.item():.2%}"
           for idx, prob in zip(top_3.indices, top_3.values)
       ]
       return "\n".join(results)
   except Exception as e:
       return f"Error processing video: {str(e)}"

iface = gr.Interface(
   fn=predict_video,
   inputs=gr.Video(label="Upload Video"),
   outputs=gr.Textbox(label="Top 3 Predictions"),
   title="Video Action Recognition",
   description="Upload a video to classify the action being performed. The model will return the top 3 predictions.",
   examples=[
       ["test_video_1.avi"],
       ["test_video_2.avi"],
       ["test_video_3.avi"]
   ],
   cache_examples=True
)

if __name__ == "__main__":
   iface.launch(
       debug=False,
       server_name="0.0.0.0",
       server_port=7860
   )