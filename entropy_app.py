import streamlit as st
import numpy as np
from skimage import io, filters, img_as_ubyte
from skimage.morphology import disk
from PIL import Image
import io as python_io

def process_image(image, radius=5, tolerance=0.1):
    """Process the uploaded image and return original and highlighted versions"""
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Ensure the image is in RGB format
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    elif len(image_array.shape) == 2:
        # Convert grayscale to RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # Split the image into RGB channels
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]
    
    # Convert channels to uint8 if necessary
    red_channel = img_as_ubyte(red_channel)
    green_channel = img_as_ubyte(green_channel)
    blue_channel = img_as_ubyte(blue_channel)
    
    # Calculate Local Entropy
    selem = disk(radius)
    
    entropy_red = filters.rank.entropy(red_channel, selem)
    entropy_green = filters.rank.entropy(green_channel, selem)
    entropy_blue = filters.rank.entropy(blue_channel, selem)
    
    # Compare Entropy Across Channels
    entropy_diff_rg = np.abs(entropy_red - entropy_green)
    entropy_diff_rb = np.abs(entropy_red - entropy_blue)
    entropy_diff_gb = np.abs(entropy_green - entropy_blue)
    
    # Create a mask where entropy differences are within the tolerance
    mask = (entropy_diff_rg < tolerance) & (entropy_diff_rb < tolerance) & (entropy_diff_gb < tolerance)
    
    # Highlight Matching Pixels
    highlighted_image = image_array.copy()
    highlighted_image[mask] = [255, 0, 0]  # Mark matching pixels in red
    
    return image_array, highlighted_image, mask, entropy_red, entropy_green, entropy_blue

def main():
    st.title("🔍 Image Entropy Analysis Tool")
    st.markdown("""
    This tool analyzes the local entropy of RGB channels in your image and highlights 
    pixels where all three channels have similar entropy values.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    radius = st.sidebar.slider("Neighborhood Radius", 1, 10, 5, 
                              help="Size of the neighborhood for entropy calculation")
    tolerance = st.sidebar.slider("Entropy Tolerance", 0.01, 1.0, 0.1, 
                                 help="Maximum difference allowed between channel entropies")
    
    # Image source selection
    st.sidebar.header("Image Source")
    image_source = st.sidebar.radio(
        "Choose image source:",
        ["Upload File", "Use Local File"]
    )
    
    image = None
    
    if image_source == "Upload File":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to analyze its entropy patterns"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    else:  # Use Local File
        # Local file path input
        st.markdown("### 📁 Local File Mode")
        default_path = st.text_input(
            "Enter image file path:", 
            value="image.jpg",
            help="Enter the path to your local image file (e.g., 'image.jpg', 'path/to/image.png')"
        )
        
        load_button = st.button("Load Local Image")
        
        if load_button and default_path.strip():
            try:
                # Step 1: Reading the Image (Original functionality)
                image_array = io.imread(default_path.strip())
                
                # Convert to PIL Image for consistency
                if len(image_array.shape) == 2:  # Grayscale
                    image = Image.fromarray(image_array, mode='L').convert('RGB')
                elif len(image_array.shape) == 3:
                    if image_array.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
                    else:  # RGB
                        image = Image.fromarray(image_array, mode='RGB')
                else:
                    st.error("Unsupported image format")
                    image = None
                    
                if image is not None:
                    st.success(f"✅ Successfully loaded: {default_path}")
                
            except FileNotFoundError:
                st.error(f"❌ File not found: '{default_path}'")
                st.info("Make sure the file path is correct and the file exists.")
                image = None
            except Exception as e:
                st.error(f"❌ Could not load image from '{default_path}': {str(e)}")
                st.info("Make sure the file exists and is a valid image format.")
                image = None
        elif load_button and not default_path.strip():
            st.warning("⚠️ Please enter a file path first.")
    
    if image is not None:
        try:
            # Verify image is properly loaded
            if not hasattr(image, 'width') or not hasattr(image, 'height'):
                st.error("Invalid image object. Please try loading the image again.")
                return
            
            # Display original image info
            st.subheader("📊 Image Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", image.width)
            with col2:
                st.metric("Height", image.height)
            with col3:
                st.metric("Mode", image.mode)
            
            # Process the image
            with st.spinner("Processing image... This may take a moment."):
                original, highlighted, mask, entropy_red, entropy_green, entropy_blue = process_image(image, radius, tolerance)
            
            # Display results
            st.subheader("🖼️ Results")
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original, use_container_width=True)
            
            with col2:
                st.markdown("**Highlighted Image**")
                st.image(highlighted, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Analysis Statistics")
            total_pixels = mask.size
            highlighted_pixels = np.sum(mask)
            percentage = (highlighted_pixels / total_pixels) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pixels", f"{total_pixels:,}")
            with col2:
                st.metric("Highlighted Pixels", f"{highlighted_pixels:,}")
            with col3:
                st.metric("Percentage Highlighted", f"{percentage:.2f}%")
            
            # Show entropy heatmaps using Streamlit's native image display
            if st.checkbox("Show Entropy Heatmaps"):
                st.subheader("🔥 Channel Entropy Heatmaps")
                st.info("Entropy values are displayed as grayscale images (brighter = higher entropy)")
                
                # Normalize entropy values to 0-255 for display
                entropy_red_norm = ((entropy_red - entropy_red.min()) / (entropy_red.max() - entropy_red.min()) * 255).astype(np.uint8)
                entropy_green_norm = ((entropy_green - entropy_green.min()) / (entropy_green.max() - entropy_green.min()) * 255).astype(np.uint8)
                entropy_blue_norm = ((entropy_blue - entropy_blue.min()) / (entropy_blue.max() - entropy_blue.min()) * 255).astype(np.uint8)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Red Channel Entropy**")
                    st.image(entropy_red_norm, use_container_width=True)
                    st.caption(f"Min: {entropy_red.min():.2f}, Max: {entropy_red.max():.2f}")
                
                with col2:
                    st.markdown("**Green Channel Entropy**")
                    st.image(entropy_green_norm, use_container_width=True)
                    st.caption(f"Min: {entropy_green.min():.2f}, Max: {entropy_green.max():.2f}")
                
                with col3:
                    st.markdown("**Blue Channel Entropy**")
                    st.image(entropy_blue_norm, use_container_width=True)
                    st.caption(f"Min: {entropy_blue.min():.2f}, Max: {entropy_blue.max():.2f}")
            
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.info("Please try uploading a different image or adjusting the parameters.")
    
    else:
        st.info("👆 Please upload an image or specify a local file path to get started!")
        
        # Show sample explanation
        with st.expander("ℹ️ How does this work?"):
            st.markdown("""
            This tool performs the following steps:
            
            1. **Split Channels**: Separates your image into Red, Green, and Blue channels
            2. **Calculate Entropy**: Computes local entropy for each channel using a sliding window
            3. **Compare Entropies**: Finds pixels where all three channels have similar entropy values
            4. **Highlight Results**: Marks matching pixels in red on the output image
            
            **Parameters:**
            - **Neighborhood Radius**: Size of the area around each pixel used for entropy calculation
            - **Entropy Tolerance**: How similar the entropy values need to be to be considered "matching"
            
            **Use Cases:**
            - Detecting uniform or textured regions
            - Image analysis and segmentation
            - Quality assessment
            """)

if __name__ == "__main__":
    main()
