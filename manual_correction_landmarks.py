from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QEventLoop
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QHBoxLayout, QSlider, QTableWidget, QTableWidgetItem, QMessageBox, QAbstractItemView, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt
import mediapipe as mp
import cv2
import pandas as pd
import sys
import queue
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import butter, filtfilt



class HandTrackingWorker(QThread):
    frame_processed = pyqtSignal(object, object, int)  # Emit frame, landmarks, and frame index
    progress = pyqtSignal(int)  # Emit progress percentage
    error = pyqtSignal(str)  # Emit errors to the main thread
    finished_tracking = pyqtSignal()  # Emit when tracking is complete

    def __init__(self, video_path, hands, result_queue, include_virtual_arm):
        super().__init__()
        self.video_path = video_path
        self.hands = hands
        self.running = True
        self.result_queue = result_queue
        self.include_virtual_arm = include_virtual_arm
    
    def low_pass_filter(self, data, cutoff=1, fs=30, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y
    
    def smooth_landmarks(self, landmarks_df):
        if landmarks_df is None or landmarks_df.empty:
            return None
            
        smoothed_data = []
        for landmark in range(22):
            landmark_data = landmarks_df[landmarks_df['landmark_index'] == landmark].copy()
            if not landmark_data.empty:
                landmark_data['x'] = self.low_pass_filter(landmark_data['x'].values)
                landmark_data['y'] = self.low_pass_filter(landmark_data['y'].values)
                landmark_data['z'] = self.low_pass_filter(landmark_data['z'].values)
                smoothed_data.append(landmark_data)
        
        return pd.concat(smoothed_data).sort_values(['Frame', 'landmark_index']).reset_index(drop=True)


    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Error opening video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            all_landmarks = []  # Collect landmarks for all frames

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    frame_landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            frame_landmarks.append({
                                "Frame": frame_idx,
                                "landmark_index": i,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z
                            })
                        # Approximate arm with offsets from x, y, z
                        if self.include_virtual_arm:
                            wrist = hand_landmarks.landmark[0]
                            frame_landmarks.append({
                                "Frame": frame_idx,
                                "landmark_index": 21,
                                "x": wrist.x + 0.02,
                                "y": wrist.y - 0.15,
                                "z": wrist.z - 0.03
                            })
                    all_landmarks.append(pd.DataFrame(frame_landmarks))

                # Emit progress
                progress = int((frame_idx / total_frames) * 100)
                self.progress.emit(progress)
                frame_idx += 1

            cap.release()
            self.progress.emit(100)

            # After all frames are processed, concatenate and smooth once
            if all_landmarks:
                full_df = pd.concat(all_landmarks, ignore_index=True)
                smoothed_df = self.smooth_landmarks(full_df)
                self.result_queue.put(smoothed_df.to_dict(orient='records'))

            self.finished_tracking.emit()

        except Exception as e:
            self.error.emit(str(e))
            # Log the error in the console
            print(f"Error in HandTrackingWorker: {e}")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class HandTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Tracking Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Add mouse tracking attributes
        self.dragging = False
        self.selected_landmark = None
        self.drag_start_pos = None

        # Initialize attributes
        self.is_editing = False
        self.video_path = None
        self.worker = None
        self.result_queue = queue.Queue()
        self.landmarks_df = pd.DataFrame()
        self.current_frame_index = 0
        self.playing = False
        self.start_timestamp = datetime.now()  # Default to current time
        self.fps = 0.0

        # Mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.mp_drawing = mp.solutions.drawing_utils

        # UI setup
        self.init_ui()

        # Timer to fetch results from the queue
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(100)  # Check every 100ms

    def init_ui(self):
        layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel("Video will be displayed here")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.load_video_button = QPushButton("Load Video")
        self.import_tracking = QPushButton("Import tracking")
        self.track_hands_button = QPushButton("Track Hands")
        self.save_button = QPushButton("Save")
        self.edit_button = QPushButton("Edit Landmarks")
        self.interpolate_button = QPushButton("Interpolate")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        button_layout.addWidget(self.load_video_button)
        button_layout.addWidget(self.import_tracking)
        button_layout.addWidget(self.track_hands_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.interpolate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        layout.addLayout(button_layout)

        # Slider for navigating frames
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        layout.addWidget(self.frame_slider)

        # Frame number display
        self.frame_number_label = QLabel("Time: NA (Frame: 0)")
        layout.addWidget(self.frame_number_label)

        # Landmark Table
        self.landmark_table = QTableWidget()
        layout.addWidget(self.landmark_table)

        # Set up connections
        self.load_video_button.clicked.connect(self.load_video)
        self.track_hands_button.clicked.connect(self.start_hand_tracking)
        self.save_button.clicked.connect(self.save_landmarks)
        self.edit_button.clicked.connect(self.toggle_editing)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.landmark_table.itemChanged.connect(self.on_table_item_changed)
        self.import_tracking.clicked.connect(self.import_tracking_func)
        self.interpolate_button.clicked.connect(self.interpolate_landmarks)

        # Enable mouse tracking for video label
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self.mouse_press_event
        self.video_label.mouseMoveEvent = self.mouse_move_event
        self.video_label.mouseReleaseEvent = self.mouse_release_event

        # Set layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def interpolate_landmarks(self):
        if self.landmarks_df.empty:
            QMessageBox.warning(self, "Warning", "No landmarks to interpolate.")
            return

        # Interpolate landmarks between 2 frames
        start_frame, ok = QInputDialog.getInt(self, "Start Frame", "Enter the start frame index:")
        if not ok:
            return
        
        end_frame, ok = QInputDialog.getInt(self, "End Frame", "Enter the end frame index:")
        if not ok:
            return
        
        if start_frame >= end_frame:
            QMessageBox.warning(self, "Warning", "Start frame must be less than end frame.")
            return
        
        # Select specific landmarks to interpolate
        landmark_indices, ok = QInputDialog.getText(self, "Landmark Indices", "Enter landmark indices to interpolate (comma-separated, leave empty for all):")
        if not ok:
            return
        
        if landmark_indices:
            landmark_indices = [int(idx.strip()) for idx in landmark_indices.split(",")]
        else:
            landmark_indices = self.landmarks_df["landmark_index"].unique()

        # Get landmarks for start and end frames
        start_landmarks = self.get_landmarks_for_frame(start_frame)
        end_landmarks = self.get_landmarks_for_frame(end_frame)

        if start_landmarks.empty or end_landmarks.empty:
            QMessageBox.warning(self, "Warning", "No landmarks found for start or end frame.")
            return
        
        # Interpolate between the two frames
        interpolated_landmarks = []
        for frame_idx in range(start_frame, end_frame + 1):  # Include start_frame and end_frame
            # Calculate interpolation factor
            alpha = (frame_idx - start_frame) / (end_frame - start_frame)
            
            # Interpolate landmarks
            for _, start_row in start_landmarks.iterrows():
                if start_row["landmark_index"] in landmark_indices:
                    end_row = end_landmarks[end_landmarks["landmark_index"] == start_row["landmark_index"]]
                    
                    if not end_row.empty:
                        # Interpolate X, Y, Z coordinates
                        x = (1 - alpha) * start_row["x"] + alpha * end_row["x"].values[0]
                        y = (1 - alpha) * start_row["y"] + alpha * end_row["y"].values[0]
                        z = (1 - alpha) * start_row["z"] + alpha * end_row["z"].values[0]
                        
                        interpolated_landmarks.append({
                            "Frame": frame_idx,
                            "landmark_index": start_row["landmark_index"],
                            "x": x,
                            "y": y,
                            "z": z
                        })

        # Create DataFrame for interpolated landmarks
        interpolated_df = pd.DataFrame(interpolated_landmarks)

        # Check for NaN values
        if interpolated_df.isna().any().any():
            print("NaN values found in interpolated DataFrame")
            print(interpolated_df[interpolated_df.isna().any(axis=1)])
        
        # Fill NaN values with interpolated values
        interpolated_df = interpolated_df.fillna(method='ffill').fillna(method='bfill')

        # Update only the interpolated landmarks in the existing DataFrame
        for idx in landmark_indices:
            mask = (self.landmarks_df["Frame"] >= start_frame) & (self.landmarks_df["Frame"] <= end_frame) & (self.landmarks_df["landmark_index"] == idx)
            self.landmarks_df = self.landmarks_df[~mask]
        
        self.landmarks_df = pd.concat([self.landmarks_df, interpolated_df], ignore_index=True)

        self.update_table()
        self.load_frame(self.current_frame_index)
        QMessageBox.information(self, "Interpolation", "Landmarks have been interpolated successfully.")
    
    def import_tracking_func(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select tracking File", "", "Csv Files (*.csv)")
        if self.video_path:
            pdframe = pd.read_csv(self.video_path)
            print(self.landmarks_df)
            # Check if the required columns are present
            required_columns = ['timestamp', 'landmark_index', 'x', 'y', 'z']
            if not all(column in pdframe.columns for column in required_columns):
                QMessageBox.warning(self, "Warning", "The selected file does not contain the required columns.")
                return
            
            self.landmarks_df = pdframe.copy()
            
            # Make the imported start time the new start time
            self.start_timestamp = datetime.strptime(self.landmarks_df['timestamp'].iloc[0], "%Y-%m-%d %H:%M:%S.%f")

            # Convert timestamp to frame index
            self.landmarks_df['Frame'] = self.landmarks_df['timestamp'].apply(
                lambda x: self.timestamp_to_frame(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            )

            # Low-pass filter the imported landmarks
            self.landmarks_df = self.smooth_landmarks(self.landmarks_df)

            # self.track_hands_button.setEnabled(False)
            self.update_table()
            self.load_frame(self.current_frame_index)
            print(self.landmarks_df)

    def low_pass_filter(self, data, cutoff=1, fs=30, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y
    
    def smooth_landmarks(self, landmarks_df):
        if landmarks_df is None or landmarks_df.empty:
            return None
            
        smoothed_data = []
        for landmark in range(22):
            landmark_data = landmarks_df[landmarks_df['landmark_index'] == landmark].copy()
            if not landmark_data.empty:
                landmark_data['x'] = self.low_pass_filter(landmark_data['x'].values)
                landmark_data['y'] = self.low_pass_filter(landmark_data['y'].values)
                landmark_data['z'] = self.low_pass_filter(landmark_data['z'].values)
                smoothed_data.append(landmark_data)
        
        return pd.concat(smoothed_data).sort_values(['Frame', 'landmark_index']).reset_index(drop=True)

    def toggle_editing(self):
        """Toggle between editing modes."""
        if self.is_editing:
            self.disable_editing()
        else:
            self.enable_editing()

    def init_video_writer(self, output_path, frame_size, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    
    def mouse_press_event(self, event):
        if not self.is_editing:
            return
            
        # Get click position relative to label
        pos = event.pos()
        frame_pos = (pos.x(), pos.y())
        
        # Get current frame landmarks
        landmarks = self.get_landmarks_for_frame(self.current_frame_index)
        if landmarks.empty:
            return
            
        # Find closest landmark within threshold
        threshold = 10  # pixels
        closest_landmark = None
        min_dist = float('inf')
        
        for _, row in landmarks.iterrows():
            x = int(row["x"] * self.video_label.width())
            y = int(row["y"] * self.video_label.height())
            dist = ((frame_pos[0] - x)**2 + (frame_pos[1] - y)**2)**0.5
            
            if dist < threshold and dist < min_dist:
                min_dist = dist
                closest_landmark = row["landmark_index"]
        
        if closest_landmark is not None:
            self.dragging = True
            self.selected_landmark = closest_landmark
            self.drag_start_pos = frame_pos

    def mouse_move_event(self, event):
        if not self.dragging or self.selected_landmark is None:
            return
            
        # Get current position
        pos = event.pos()
        
        # Convert to normalized coordinates
        x = pos.x() / self.video_label.width()
        y = pos.y() / self.video_label.height()
        
        # Update DataFrame
        mask = (self.landmarks_df["Frame"] == self.current_frame_index) & \
                (self.landmarks_df["landmark_index"] == self.selected_landmark)
        
        self.landmarks_df.loc[mask, "x"] = x
        self.landmarks_df.loc[mask, "y"] = y
        
        # Redraw frame
        self.load_frame(self.current_frame_index)
        
        # Update table
        self.update_table()

    def mouse_release_event(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_landmark = None
            self.drag_start_pos = None

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file.")
                return
                
            # Get video FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Get starting timestamp from user
            timestamp_str, ok = QInputDialog.getText(
                self,
                "Enter Starting Timestamp",
                "Enter the starting timestamp (YYYY-MM-DD HH:MM:SS.ffffff):",
                text=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            )
            if ok:
                try:
                    self.start_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    QMessageBox.critical(self, "Error", "Invalid timestamp format. Using current time.")
                    self.start_timestamp = datetime.now()
                    
            self.frame_slider.setEnabled(True)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.load_frame(0)

    def frame_to_timestamp(self, frame_idx):
        """Convert frame index to timestamp."""
        seconds = frame_idx / self.fps
        return self.start_timestamp + timedelta(seconds=seconds)

    def timestamp_to_frame(self, timestamp):
        """Convert timestamp to nearest frame index."""
        delta = timestamp - self.start_timestamp
        return round(delta.total_seconds() * self.fps)

    def load_frame(self, index):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = self.cap.read()
            if success:
                self.current_frame_index = index
                landmarks = self.get_landmarks_for_frame(index)
                self.display_frame(frame, landmarks)

    def display_frame(self, frame, landmarks=None):
        # Convert frame to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Load and add MediaPipe hand landmark reference image
        try:
            legend_img = cv2.imread("Code\mediapipe\manual_correction_util\mediapipe_manual_correction\MediaPipe-Hands-21-landmarks-13.png", cv2.IMREAD_UNCHANGED)
            if legend_img is not None:
                # Scale legend image (adjust size as needed)
                legend_height = 150  # Desired height in pixels
                aspect_ratio = legend_img.shape[1] / legend_img.shape[0]
                legend_width = int(legend_height * aspect_ratio)
                legend_img = cv2.resize(legend_img, (legend_width, legend_height))
                
                # Position in top-right corner with padding
                padding = 10
                y_offset = padding
                x_offset = width - legend_width - padding
                
                # Create ROI for legend
                roi = rgb_image[y_offset:y_offset + legend_height, 
                            x_offset:x_offset + legend_width]
                
                # Handle transparency if PNG
                if legend_img.shape[2] == 4:  # With alpha channel
                    alpha = legend_img[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * legend_img[:, :, c]
                else:  # Without alpha
                    rgb_image[y_offset:y_offset + legend_height,
                            x_offset:x_offset + legend_width] = legend_img[:, :, :3]
                    
        except Exception as e:
            print(f"Could not load legend image: {e}")

        
        # Define coordinate system origin point (bottom-left corner, with some padding)
        origin = (50, height - 50)
        axis_length = 100  # Length of axis lines in pixels
        
        # Draw coordinate system
        # X-axis (Red)
        cv2.arrowedLine(rgb_image, origin, 
                        (origin[0] + axis_length, origin[1]), 
                        (255, 0, 0), 2, tipLength=0.2)
        cv2.putText(rgb_image, "X", 
                    (origin[0] + axis_length + 10, origin[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Y-axis (Green) - Remember Y is inverted in image coordinates
        cv2.arrowedLine(rgb_image, origin,
                        (origin[0], origin[1] - axis_length),
                        (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(rgb_image, "Y Inverted",
                    (origin[0] - 20, origin[1] - axis_length - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Z-axis (Blue) - Drawn at 45 degrees to indicate depth
        z_end = (origin[0] + int(axis_length * 0.7),
                origin[1] - int(axis_length * 0.7))
        cv2.arrowedLine(rgb_image, origin, z_end,
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(rgb_image, "Z",
                    (z_end[0] + 10, z_end[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw landmarks if available
        if landmarks is not None and not landmarks.empty:
            landmark_points = {}
            
            # First draw connection lines (behind dots)
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # Index finger
                (0, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17),  # Metacarpal connections
                # connect wrist to virtual arm
                (0, 21),
            ]

            # Draw larger hit areas first (semi-transparent)
            for _, row in landmarks.iterrows():
                    x = int(row["x"] * frame.shape[1])
                    y = int(row["y"] * frame.shape[0])
                    landmark_points[row["landmark_index"]] = (x, y)
                    cv2.circle(rgb_image, (x, y), 6, (255, 255, 255, 128), -1)


            # Draw connections
            for start_idx, end_idx in hand_connections:
                if start_idx in landmark_points and end_idx in landmark_points:
                    cv2.line(rgb_image, 
                            landmark_points[start_idx],
                            landmark_points[end_idx],
                            (0, 255, 0), 2)

            # Draw landmark dots on top
            for _, row in landmarks.iterrows():
                x = int(row["x"] * frame.shape[1])
                y = int(row["y"] * frame.shape[0])
                # Draw smaller solid circle for actual landmark
                cv2.circle(rgb_image, (x, y), 8, (255, 0, 0), -1)
                # Add white outline
                cv2.circle(rgb_image, (x, y), 8, (255, 255, 255), 1)

            # Highlight selected/dragged landmark
            if self.dragging and self.selected_landmark is not None:
                    mask = (landmarks["landmark_index"] == self.selected_landmark)
                    if any(mask):
                        x = int(landmarks[mask]["x"].values[0] * frame.shape[1])
                        y = int(landmarks[mask]["y"].values[0] * frame.shape[0])

                    # Draw prominent highlight
                    cv2.circle(rgb_image, (x, y), 20, (0, 255, 255), 2)
                    cv2.circle(rgb_image, (x, y), 12, (0, 255, 255), -1)

        # Convert to QImage for display
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update QLabel
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # Write frame to video if recording
        if hasattr(self, 'video_writer') and self.video_writer.isOpened():
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_image)

        # Add visual feedback for selected landmark
        if self.dragging and self.selected_landmark is not None:
            mask = (landmarks["landmark_index"] == self.selected_landmark)
            if any(mask):
                x = int(landmarks[mask]["x"].values[0] * frame.shape[1])
                y = int(landmarks[mask]["y"].values[0] * frame.shape[0])
                # Draw highlight circle around selected landmark
                cv2.circle(rgb_image, (x, y), 8, (0, 255, 255), 2)


    def start_hand_tracking(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return
        
        answer, ok = QInputDialog.getText(self, "Include Virtual Arm?", "Include virtual arm? (yes/no):", text="yes")
        include_virtual_arm = ok and answer.lower().startswith("y")

        # Stop existing worker if running
        if self.worker:
            self.worker.stop()

        # Clear existing data
        self.landmarks_df = pd.DataFrame()
        self.worker = HandTrackingWorker(self.video_path, self.hands, self.result_queue, include_virtual_arm)

        # Start worker
        self.worker = HandTrackingWorker(self.video_path, self.hands, self.result_queue, include_virtual_arm)
        self.worker.frame_processed.connect(self.on_frame_processed)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_worker_error)
        self.worker.finished_tracking.connect(self.on_tracking_finished)  # Connect the finished signal
        self.worker.start()

    def on_frame_processed(self, frame, landmarks, frame_idx):
        # Handle the frame and landmarks (process the result and update the UI)
        # Display the landmarks on the frame (for example, drawing circles on landmarks)
        self.display_frame(frame, landmarks)

        # Optionally, update the slider or other parts of the UI if necessary
        self.frame_slider.setValue(frame_idx)

        # If you want to do any additional processing with the landmarks, you can store or update them here
        landmarks_df = pd.DataFrame(landmarks)
        self.landmarks_df = pd.concat([self.landmarks_df, landmarks_df], ignore_index=True)

        # Update the table with the latest landmarks for the current frame
        self.update_table()

    def process_queue(self):
        while not self.result_queue.empty():
            frame_landmarks = self.result_queue.get()
            new_df = pd.DataFrame(frame_landmarks)
            self.landmarks_df = pd.concat([self.landmarks_df, new_df], ignore_index=True)

        # Update the table with the current frame's landmarks
        self.update_table()

    def on_frame_change(self, value):
        """Handle slider value change."""
        # Update current frame index
        self.current_frame_index = value
        
        # Load and display the frame
        self.load_frame(value)
        
        # Update timestamp display
        timestamp = self.frame_to_timestamp(value)
        self.frame_number_label.setText(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} (Frame: {value})")
        
        # Force table update with current frame's landmarks
        landmarks = self.get_landmarks_for_frame(value)
        if not landmarks.empty:
            self.update_table()

    def get_landmarks_for_frame(self, frame_idx):
        """Get landmarks for specific frame."""
        if "Frame" not in self.landmarks_df.columns or self.landmarks_df.empty:
            return pd.DataFrame()
        
        return self.landmarks_df[self.landmarks_df["Frame"] == frame_idx]

    def update_table(self):
        """Updates the table with the current frame's landmarks."""
        if self.is_editing:  # Skip table refresh if editing is active
            return

        self.landmark_table.blockSignals(True)
        
        # Get landmarks for current frame
        landmarks = self.get_landmarks_for_frame(self.current_frame_index)
        
        # Update table
        self.landmark_table.setRowCount(len(landmarks))
        self.landmark_table.setColumnCount(5)
        self.landmark_table.setHorizontalHeaderLabels(["timestamp", "landmark_index", "x", "y", "z"])

        for i, (_, row) in enumerate(landmarks.iterrows()):
            # Calculate timestamp for current frame
            timestamp = self.frame_to_timestamp(row["Frame"])
            
            # Timestamp column
            item = QTableWidgetItem(timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.landmark_table.setItem(i, 0, item)
            
            # Other columns
            for j, key in enumerate(["landmark_index", "x", "y", "z"], 1):
                item = QTableWidgetItem(str(row[key]))
                if key in ["x", "y", "z"]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.landmark_table.setItem(i, j, item)

        self.landmark_table.blockSignals(False)

    def on_table_item_changed(self, item):
        """Handles updates to table cells."""
        if not self.is_editing:
            return

        try:
            row = item.row()
            col = item.column()
            
            # Only process X, Y, Z columns (index 2, 3, 4)
            if col < 2:  # Skip Frame and Landmark columns
                return
                    
            # Get input value and clean it
            input_text = item.text().strip().replace(',', '.')  # Replace comma with decimal point
            if not input_text:  # Skip empty values
                return

            # Print debug info
            print(f"Input text: '{input_text}'")
                    
            try:
                value = float(input_text)
                print(f"Converted value: {value}")
            except ValueError:
                QMessageBox.critical(self, "Invalid Input", f"'{input_text}' is not a valid number. Please enter a number like 0.5 or -1.23")
                item.setText("")
                return
                
            # Block signals to prevent recursive calls
            self.landmark_table.blockSignals(True)

            # Update DataFrame - Convert frame and landmark indices properly
            timestamp_str = self.landmark_table.item(row, 0).text()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            frame_idx = self.timestamp_to_frame(timestamp)
        
            landmark_idx = int(float(self.landmark_table.item(row, 1).text()))
            column_name = self.landmark_table.horizontalHeaderItem(col).text()

            # Update the value
            self.landmarks_df.loc[
                (self.landmarks_df["Frame"] == frame_idx) & (self.landmarks_df["landmark_index"] == landmark_idx),
                column_name
            ] = value

            # Format display value
            formatted_value = f"{value:.17f}"
            print(f"Formatted value: {formatted_value}")
            item.setText(formatted_value)
            
            # Unblock signals
            self.landmark_table.blockSignals(False)
                
        except Exception as e:
            print(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            item.setText("")
            self.landmark_table.blockSignals(False)

    def save_landmarks(self):
        # Save landmarks
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Landmarks", "", "CSV Files (*.csv)")
        if save_path:
            try:
                df_to_save = self.landmarks_df.copy()
                df_to_save = df_to_save.sort_values(["Frame", "landmark_index"])
                df_to_save['timestamp'] = df_to_save['Frame'].apply(
                    lambda x: self.frame_to_timestamp(x).strftime("%Y-%m-%d %H:%M:%S.%f")
                )
                columns = ['timestamp', 'landmark_index', 'x', 'y', 'z']
                df_to_save = df_to_save[columns]
                df_to_save.to_csv(save_path, index=False)
                QMessageBox.information(self, "Success", "Landmarks saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save landmarks: {e}")

        # Save video
        video_output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4);;AVI Files (*.avi)")
        if video_output_path:
            try:
                frame_size = (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                self.init_video_writer(video_output_path, frame_size, self.fps)

                for i in range(self.total_frames):
                    self.load_frame(i)  # Writes each frame
                    # Indicate saving progress
                    percentage = int((i / self.total_frames) * 100)
                    self.update_progress(percentage)

                QMessageBox.information(self, "Success", "Video saved successfully!")
                self.video_writer.release()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save video: {e}")


    def update_progress(self, progress):
        self.setWindowTitle(f"Hand Tracking Editor - Progress: {progress}%")

    def handle_worker_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def on_tracking_finished(self):
        print(self.landmarks_df)
        QMessageBox.information(self, "Info", "Hand tracking is complete. You can now edit the landmarks.")
        # Enable editing triggers using correct enum values
        self.landmark_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | 
            QAbstractItemView.EditKeyPressed |
            QAbstractItemView.AnyKeyPressed
        )
        # Update table and load first frame
        self.update_table()
        self.load_frame(0)
        self.on_frame_change(0)

    def enable_editing(self):
        """Enables editing mode and disables navigation controls."""
        self.is_editing = True
        self.edit_button.setText("Done Editing")
        
        # Disable navigation controls
        self.frame_slider.setEnabled(False)
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        
        QMessageBox.information(self, "Edit Mode", "You can now edit landmarks. Click 'Done Editing' when finished.")

    def disable_editing(self):
        """Disables editing mode and re-enables navigation controls."""
        self.is_editing = False
        self.edit_button.setText("Edit Landmarks")
        
        # Re-enable navigation controls
        self.frame_slider.setEnabled(True)
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        
        self.update_table()  # Force table update
        QMessageBox.information(self, "Edit Mode", "Editing mode disabled. Changes have been saved.")



    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.cap:
            self.cap.release()
        self.hands.close()
        event.accept()

    def play_video(self):
        self.playing = True
        self.play_video_loop()

    def pause_video(self):
        self.playing = False

    def play_video_loop(self):
        if self.playing and self.current_frame_index < self.total_frames:
            self.current_frame_index += 1
            self.frame_slider.setValue(self.current_frame_index)
            QEventLoop().processEvents()
            QTimer.singleShot(30, self.play_video_loop)  # Adjust the delay as needed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandTrackingApp()
    window.show()
    sys.exit(app.exec_())