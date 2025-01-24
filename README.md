# Handwritten Digit Recognition

A Python application that recognizes handwritten digits using RandomForest classifier.

## Features
- Real-time digit recognition
- Drawing interface using Pygame
- Machine learning using scikit-learn
- Confidence scores for predictions

## Setup
1. Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

2. Install required packages
    pip install scikit-learn numpy pygame Pillow matplotlib

3. Run program
    python classifier.py

## Usage
- Draw digit with your mouse
- Press Enter to display prediction
- Press 'c' to clear
- Close window to exit

## Requirements
- Python 3.x
- Terminal/Command line
- Basic graphics support

## Description
This application uses machine learning to recognize hand-drawn digits (0-9). It employs a Random Forest classifier trained on the MNIST dataset, providing real-time predictions with confidence scores.
