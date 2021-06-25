import time
import cv2
import face_recognition
import numpy as np
import pickle

video_capture = cv2.VideoCapture(0)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

known_face_encodings = []
known_face_names = []


def current_sec_time():
    return round(time.time() * 1000)


print("########################################")
start_time = current_sec_time();

# Load face encodings
with open('assets/dataset_faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)
known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

end_time = current_sec_time()
print("init time " + str((end_time - start_time) / 1000) + "s")
print("########################################")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        start_time = current_sec_time();
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            score = 0
            for value in face_distances:
                score = score + value
            score = score / 23
            score = round(score, 2) * 100
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 18), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + " " + str(int(score)) + "%", (left + 6, bottom - 3), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Who am I?', frame)
    end_time = current_sec_time()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
