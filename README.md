# Mediapipe Manual Tracking Editor

## Overview

The Mediapipe Manual Tracking Editor is a GUI application built with PyQt5 for tracking and editing hand landmarks in video files. It uses MediaPipe for hand tracking and allows users to manually correct and interpolate landmarks. For now it currently only supports [hand_landmarkers detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

## Features

- Load and display video files
- Track hand landmarks using MediaPipe
- Edit hand landmarks manually
- Interpolate landmarks between frames
- Import and export landmark data in CSV format
- Save annotated video with landmarks

## Requirements

- Python 3.6+
- PyQt5
- OpenCV
- MediaPipe
- Pandas
- NumPy
- SciPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/firmanserdana/mediapipe_manual_correction
   cd mediapipe_manual_correction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python manual_correction_landmarks.py
   ```

2. Load a video file by clicking the "Load Video" button.

3. Track hand landmarks by clicking the "Track Hands" button.

4. Edit landmarks by clicking the "Edit Landmarks" button. Click on a landmark to select it and drag to move it.

5. Interpolate landmarks between frames by clicking the "Interpolate" button and entering the start and end frames.

6. Save the landmarks and annotated video by clicking the "Save" button.

7. Import existing tracking data by clicking the "Import tracking" button and selecting a CSV file.

## Controls

- **Load Video**: Load a video file for processing.
- **Track Hands**: Start tracking hand landmarks in the loaded video.
- **Edit Landmarks**: Enable manual editing of landmarks.
- **Interpolate**: Interpolate landmarks between two frames.
- **Save**: Save the landmarks to a CSV file and the annotated video.
- **Play**: Play the video.
- **Pause**: Pause the video.

## Editing Landmarks

1. Click the "Edit Landmarks" button to enable editing mode.
2. Click on a landmark to select it.
3. Drag the selected landmark to the desired position.
4. Click "Done Editing" to save the changes and exit editing mode.

## Interpolating Landmarks

1. Click the "Interpolate" button.
2. Enter the start and end frame indices.
3. Enter the landmark indices to interpolate (comma-separated) or leave empty to interpolate all landmarks.
4. Click "OK" to perform the interpolation.

## Importing Tracking Data

1. Click the "Import tracking" button.
2. Select a CSV file containing the tracking data.
3. The application will load and display the imported landmarks.

## Saving Landmarks and Video

1. Click the "Save" button.
2. Select a location to save the landmarks CSV file.
3. Select a location to save the annotated video file.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro) for the GUI framework
- [OpenCV](https://opencv.org/) for video processing

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the author at firmanisma.serdana@santannapisa.it / firman.serdana18@alumni.imperial.ac.uk.
