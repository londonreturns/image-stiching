import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk


class CameraCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Capture and Stitching")

        self.root.state('zoomed')

        self.capture = cv2.VideoCapture(0)
        self.image1 = None
        self.image2 = None

        self.top_frame = tk.Frame(root)
        self.top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.center_frame = tk.Frame(self.top_frame)
        self.center_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.top_frame.grid_columnconfigure(0, weight=1)

        self.video_source_label = Label(self.center_frame)
        self.video_source_label.grid(row=0, column=0, padx=5, pady=5)

        button_frame = tk.Frame(self.center_frame)
        button_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ns")

        self.capture_button = Button(button_frame, text="Capture Image 1", command=self.capture_image1)
        self.capture_button.pack(fill=tk.X, padx=5, pady=5)

        self.capture_button2 = Button(button_frame, text="Capture Image 2", command=self.capture_image2)
        self.capture_button2.pack(fill=tk.X, padx=5, pady=5)

        self.stitch_button = Button(button_frame, text="Stitch Images", command=self.stitch_images)
        self.stitch_button.pack(fill=tk.X, padx=5, pady=5)

        self.top_frame.grid_rowconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)

        self.stitched_image_label = Label(self.bottom_frame)
        self.stitched_image_label.pack(fill=tk.BOTH, expand=True)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_source_label.imgtk = imgtk
            self.video_source_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_image1(self):
        self.image1 = self.current_frame
        print("Image 1 captured")

    def capture_image2(self):
        self.image2 = self.current_frame
        print("Image 2 captured")

    def stitch_images(self):
        if self.image1 is None or self.image2 is None:
            print("Please capture both images before stitching.")
            return

        img1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=10)

        good = [m for m in matches if m[0].distance < 0.5 * m[1].distance]
        matches = np.asarray(good)

        if len(matches) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            dst_img = cv2.warpPerspective(self.image1, H,
                                          (self.image2.shape[1] + self.image1.shape[1], self.image2.shape[0]))
            dst_img[0:self.image2.shape[0], 0:self.image2.shape[1]] = self.image2

            cv2.imwrite("output.jpg", dst_img)
            img = Image.fromarray(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.stitched_image_label.imgtk = imgtk
            self.stitched_image_label.configure(image=imgtk)
            cv2.imwrite("stitched_output.jpg", dst)
        else:
            print("Not enough matches found to compute homography.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCaptureApp(root)
    root.mainloop()
