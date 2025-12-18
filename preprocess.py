from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
import os
from flask_cors import CORS
import fitz  # PyMuPDF
from datetime import datetime
from typing import Union
from pathlib import Path
from werkzeug.utils import secure_filename
import re
def check_and_correct_skew(image_path):
    """Check if image needs deskewing and apply if needed"""
    print("Step 1: Checking skew...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            
            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(median_angle) > 0.5:
                print(f"  Skew detected: {median_angle:.2f} degrees - CORRECTING")
                
                # Get image dimensions
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                
                # Perform rotation
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                corrected = cv2.warpAffine(img, M, (w, h), 
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
                
                output_path = os.path.join(OUTPUT_FOLDER, "1_deskewed.jpg")
                cv2.imwrite(output_path, corrected)
                print(f"  ✓ Saved: {output_path}")
                return output_path
    
    print("  No significant skew detected")
    output_path = os.path.join(OUTPUT_FOLDER, "1_deskewed.jpg")
    cv2.imwrite(output_path, img)
    return output_path


def detect_rotation_angle(image_path):
    """Detect if image needs 90/180/270 degree rotation with robust logic"""
    print("Step 2: Checking rotation...")
    
    img = cv2.imread(image_path)
    
    # Try all 4 orientations and get OCR results
    orientations = [0, 90, 180, 270]
    orientation_scores = {}
    
    for angle in orientations:
        # Rotate image
        if angle == 0:
            rotated = img.copy()
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:  # 270
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Save temporarily
        temp_path = os.path.join(OUTPUT_FOLDER, f"temp_rotation_{angle}.jpg")
        cv2.imwrite(temp_path, rotated)
        
        # Run OCR to get confidence and text count
        try:
            result = ocr.predict(temp_path)
            total_confidence = 0
            text_count = 0
            high_confidence_count = 0  # >0.7
            very_high_confidence_count = 0  # >0.9
            ultra_high_confidence_count = 0  # >0.95
            confidence_list = []
            
            if result and len(result) > 0:
                for page_result in result:
                    if isinstance(page_result, dict):
                        rec_texts = page_result.get('rec_texts', [])
                        rec_scores = page_result.get('rec_scores', [])
                        if rec_scores:
                            confidence_list = list(rec_scores)
                            total_confidence = sum(rec_scores)
                            text_count = len(rec_scores)
                            high_confidence_count = sum(1 for s in rec_scores if s > 0.7)
                            very_high_confidence_count = sum(1 for s in rec_scores if s > 0.9)
                            ultra_high_confidence_count = sum(1 for s in rec_scores if s > 0.95)
            
            avg_confidence = total_confidence / text_count if text_count > 0 else 0
            
            # Enhanced weighted score
            base_score = avg_confidence * 50  # Base confidence (0-50 points)
            text_count_score = min(text_count * 0.5, 20)  # Text count (0-20 points)
            high_conf_score = high_confidence_count * 1.5  # High conf bonus
            very_high_conf_score = very_high_confidence_count * 3  # Very high conf bonus
            ultra_high_conf_score = ultra_high_confidence_count * 5  # Ultra high conf bonus
            
            combined_score = (base_score + text_count_score + high_conf_score + 
                            very_high_conf_score + ultra_high_conf_score)
            
            # Consistency bonus
            if confidence_list and len(confidence_list) > 3:
                conf_std = np.std(confidence_list)
                if conf_std < 0.15:  # Very consistent
                    combined_score *= 1.1
            else:
                conf_std = 1.0
            
            orientation_scores[angle] = {
                'avg_confidence': avg_confidence,
                'text_count': text_count,
                'high_conf_count': high_confidence_count,
                'very_high_conf_count': very_high_confidence_count,
                'ultra_high_conf_count': ultra_high_confidence_count,
                'conf_std': conf_std if confidence_list else 1.0,
                'combined_score': combined_score,
                'confidence_list': confidence_list,
                'total_confidence': total_confidence
            }
            
            print(f"  Angle {angle:3d}°: avg_conf={avg_confidence:.3f}, texts={text_count:3d}, "
                  f"high={high_confidence_count:2d}, v_high={very_high_confidence_count:2d}, "
                  f"score={combined_score:.1f}")
            
        except Exception as e:
            print(f"  Error testing angle {angle}: {e}")
            orientation_scores[angle] = {
                'avg_confidence': 0,
                'text_count': 0,
                'high_conf_count': 0,
                'very_high_conf_count': 0,
                'ultra_high_conf_count': 0,
                'conf_std': 1.0,
                'combined_score': 0,
                'confidence_list': [],
                'total_confidence': 0
            }
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Find best angle
    best_angle = max(orientation_scores, key=lambda k: orientation_scores[k]['combined_score'])
    best_score = orientation_scores[best_angle]['combined_score']
    zero_score = orientation_scores[0]['combined_score']
    
    # Get second best for comparison
    sorted_angles = sorted(orientation_scores.items(), 
                          key=lambda x: x[1]['combined_score'], 
                          reverse=True)
    second_best_angle = sorted_angles[1][0] if len(sorted_angles) > 1 else 0
    second_best_score = sorted_angles[1][1]['combined_score'] if len(sorted_angles) > 1 else 0
    
    print(f"\n  Ranking: 1st={best_angle}° ({best_score:.1f}), "
          f"2nd={second_best_angle}° ({second_best_score:.1f})")
    print(f"  0° score: {zero_score:.1f}")
    
    # Decision logic with multiple criteria
    should_rotate = False
    rotation_reason = ""
    
    if best_angle == 0:
        rotation_reason = "0° is already the best orientation"
    else:
        # Calculate metrics
        score_diff = best_score - zero_score
        score_ratio = best_score / zero_score if zero_score > 0 else float('inf')
        
        # Criterion 1: 0° score is very poor (likely wrong orientation)
        if zero_score < 10:
            should_rotate = True
            rotation_reason = f"0° has very poor score ({zero_score:.1f})"
        
        # Criterion 2: Best angle score is significantly better (2x or 50+ points better)
        elif score_ratio >= 2.0:
            should_rotate = True
            rotation_reason = f"best angle {score_ratio:.1f}x better than 0°"
        
        # Criterion 3: Absolute score difference is large
        elif score_diff >= 30:
            should_rotate = True
            rotation_reason = f"score difference of {score_diff:.1f} points"
        
        # Criterion 4: Best angle has many high-confidence detections, 0° doesn't
        elif (orientation_scores[best_angle]['very_high_conf_count'] >= 3 and 
              orientation_scores[0]['very_high_conf_count'] == 0):
            should_rotate = True
            rotation_reason = "best angle has high-confidence text, 0° doesn't"
        
        # Criterion 5: Moderate improvement (1.5x) with reasonable text count
        elif score_ratio >= 1.5 and orientation_scores[best_angle]['text_count'] >= 5:
            should_rotate = True
            rotation_reason = f"1.5x+ improvement with good text detection"
        
        # Criterion 6: Best angle has much better average confidence
        elif (orientation_scores[best_angle]['avg_confidence'] > 0.75 and 
              orientation_scores[0]['avg_confidence'] < 0.5):
            should_rotate = True
            rotation_reason = f"much better avg confidence ({orientation_scores[best_angle]['avg_confidence']:.2f} vs {orientation_scores[0]['avg_confidence']:.2f})"
        
        # Criterion 7: Clear winner with gap to second place
        elif score_diff >= 20 and (best_score - second_best_score) >= 15:
            should_rotate = True
            rotation_reason = f"clear winner with {score_diff:.1f} point lead"
    
    if should_rotate:
        print(f"  ✓ ROTATING to {best_angle}°: {rotation_reason}")
    else:
        if best_angle != 0:
            print(f"  ✗ NO ROTATION: {rotation_reason if rotation_reason else 'improvement not significant enough'}")
            print(f"     (Best: {best_angle}° with {best_score:.1f}, but keeping 0° with {zero_score:.1f})")
        else:
            print(f"  ✓ NO ROTATION NEEDED: {rotation_reason}")
    
    # Apply rotation if needed
    if should_rotate:
        if best_angle == 90:
            corrected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            corrected = cv2.rotate(img, cv2.ROTATE_180)
        elif best_angle == 270:
            corrected = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            corrected = img
    else:
        corrected = img
    
    output_path = os.path.join(OUTPUT_FOLDER, "2_rotated.jpg")
    cv2.imwrite(output_path, corrected)
    print(f"  ✓ Saved: {output_path}")
    return output_path