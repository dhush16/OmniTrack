import cv2
import os

os.makedirs("data", exist_ok=True)

filename = input("Enter video filename (e.g., trial_1_normal.mp4): ").strip()
output_path = os.path.join("data", filename)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))

print(f"\nRecording to '{output_path}'.")
print("Press 'q' to stop recording.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.putText(frame, f"Recording: {filename} (Press 'q' to stop)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("OmniTrack Video Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved: {output_path}")