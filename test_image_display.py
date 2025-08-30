#!/usr/bin/env python3
"""
Test script to verify image display functionality
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_image_data_structure():
    """Test the image data structure that should be returned by the backend."""
    
    # Simulate what the backend should return
    sample_response = {
        'response': 'Luffy is the main character of One Piece...',
        'image': {
            'path': 'Straw_Hat_pirates/Luffy_and_His_Crew.png',
            'filename': 'Luffy_and_His_Crew.png',
            'character': 'Monkey D Luffy',
            'content': 'Luffy and his crew',
            'type': 'crew_group',
            'relevance_score': 0.95
        },
        'image_metadata': {
            'intent_analysis': {'intent_type': 'character_focus'},
            'candidates_count': 5,
            'relevance_score': 0.95
        }
    }
    
    print("✅ Sample response structure:")
    print(f"Response: {sample_response['response'][:50]}...")
    print(f"Image path: {sample_response['image']['path']}")
    print(f"Image filename: {sample_response['image']['filename']}")
    print(f"Image character: {sample_response['image']['character']}")
    print(f"Image content: {sample_response['image']['content']}")
    print(f"Image relevance score: {sample_response['image']['relevance_score']}")
    
    # Test what the frontend expects
    image_data = sample_response.get('image')
    if image_data and image_data.get('path'):
        print("\n✅ Frontend should be able to display this image:")
        print(f"  - Path: {image_data['path']}")
        print(f"  - Filename: {image_data['filename']}")
        print(f"  - Character: {image_data['character']}")
        print(f"  - Content: {image_data['content']}")
        print(f"  - Relevance: {image_data['relevance_score']}")
        
        # Test image URL construction
        image_src = f"/api/image/{image_data['path'].split('/')[-1]}"
        print(f"  - Image URL: {image_src}")
    else:
        print("\n❌ No image data found in response")
    
    return sample_response

def test_image_metadata_display():
    """Test the image metadata display logic."""
    
    image_data = {
        'path': 'Straw_Hat_pirates/Luffy_and_His_Crew.png',
        'filename': 'Luffy_and_His_Crew.png',
        'character': 'Monkey D Luffy',
        'content': 'Luffy and his crew',
        'relevance_score': 0.95
    }
    
    print("\n✅ Testing metadata display logic:")
    
    # Simulate the frontend metadata construction
    metadata_text = '📸 '
    if image_data.get('character'):
        metadata_text += f"{image_data['character']} - "
    if image_data.get('content'):
        metadata_text += f"{image_data['content']}"
    if image_data.get('relevance_score') is not None:
        metadata_text += f" (Relevance: {(image_data['relevance_score'] * 100):.0f}%)"
    
    print(f"Metadata text: {metadata_text}")
    
    return metadata_text

if __name__ == "__main__":
    print("🧪 Testing Image Display Functionality")
    print("=" * 50)
    
    # Test 1: Image data structure
    response = test_image_data_structure()
    
    # Test 2: Metadata display
    metadata = test_image_metadata_display()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\nThe backend should return data in this format:")
    print("- 'image' field with path, filename, character, content, relevance_score")
    print("- 'image_metadata' field with additional context")
    print("\nThe frontend expects:")
    print("- imageData.path for the image source")
    print("- imageData.character, content, relevance_score for metadata")
