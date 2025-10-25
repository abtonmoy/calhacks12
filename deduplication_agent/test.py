# After processing files...
import pipeline1

# Get all stored frames
frames = pipeline1.get_all_stored_frames()
print(f"Total stored: {len(frames)}")

# Retrieve and display a frame
frame_id = frames[0]['id']
img_array = pipeline1.get_frame_by_id(frame_id)

# Convert to PIL and display
from PIL import Image
img = Image.fromarray(img_array)
img.show()

# Find similar frames
similar_frames = pipeline1.search_similar_frames(img_array, top_k=5)
for sim in similar_frames:
    print(f"Similarity: {sim['similarity']:.3f} | {sim['source_path']}")