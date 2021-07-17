import cv2


# Object Detection using Haar feature-based cascade classifiers
# is a machine learning based approach where a cascade function
# is trained from a lot of positive and negative images.

# There are several Haar Cascades to detect multiple things already stored in the OpenCV project.
# The following XML file contains (I guess) a pretrained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# The following method returns a VideoCapture object.
# The argument of the method is either the device index or the name of a video file. 
# Normally one camera is connected. So I pass 0
video = cv2.VideoCapture(0)


if not video.isOpened():
    print("Cannot open camera")
    exit()


while True:
    # Capture the frames one after another
    ret, frame = video.read()

    # The frame variable is a numpy array which represents a bitmap using the BGR format.
    # We convert it to a gray bitmap
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # The classifier detects all the faces that are present in the gray_frame
    faces = classifier.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    magenta = (255, 0, 255)
    # Draw a rectangle around every face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), magenta, 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # We break the loop only(fans xD just kidding) when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Before ending the program release the resources and destroy all windows
video.release()
cv2.destroyAllWindows()
