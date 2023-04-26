import argparse
from face_extractor import FaceExtractor

parser = argparse.ArgumentParser(description='Extract faces from videos')
parser.add_argument('video_path', type=str, help='Video directory')
parser.add_argument('output_path', type=str, help='Output directory for face images')
parser.add_argument('--frame_count', type=int, default=30,
                    help='Number of frames to extract face')
parser.add_argument('--predictor_path', type=str, default='./shape_predictor_68_face_landmarks.dat',
                    help='Path to unzipped predictor path')
parser.add_argument('--black_width', type=int, default=89,
                    help='Width of black lines in movie')

if __name__ == "__main__":
    args = parser.parse_args()

    save_name = 'face'
    video_path = args.video_path
    output_path = args.output_path
    predictor_path = args.predictor_path
    frame_count = args.frame_count
    black_width = args.black_width

    face_extractor = FaceExtractor(save_name, video_path, output_path, predictor_path, frame_count, black_width)
    face_extractor.run()
