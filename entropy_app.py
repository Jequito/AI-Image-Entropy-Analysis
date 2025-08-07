import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    
    return image_array, highlighted_image, mask

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
        if image is not None:
            image = Image.open(uploaded_file)
    
    else:  # Use Local File
        # Local file path input
        st.markdown("### 📁 Local File Mode")
        default_path = st.text_input(
            "Enter image file path:", 
            value="image.jpg",
            help="Enter the path to your local image file (e.g., 'image.jpg', 'path/to/image.png')"
        )
        
        if st.button("Load Local Image") or default_path:
            try:
                # Step 1: Reading the Image (Original functionality)
                image_array = io.imread(default_path)
                
                # Convert to PIL Image for consistency
                if len(image_array.shape) == 2:  # Grayscale
                    image = Image.fromarray(image_array, mode='L').convert('RGB')
                elif image_array.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image_array, mode='RGBA').convert('RGB')
                else:  # RGB
                    image = Image.fromarray(image_array, mode='RGB')
                    
                st.success(f"✅ Successfully loaded: {default_path}")
                
            except Exception as e:
                st.error(f"❌ Could not load image from '{default_path}': {str(e)}")
                st.info("Make sure the file exists and is a valid image format.")
    
    if uploaded_file is not None:
        try:
            # Load the image (already loaded above based on source)
            
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
                original, highlighted, mask = process_image(image, radius, tolerance)
            
            # Display results
            st.subheader("🖼️ Results")
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original, use_column_width=True)
            
            with col2:
                st.markdown("**Highlighted Image**")
                st.image(highlighted, use_column_width=True)
            
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
            
            # Optional: Show entropy visualization
            if st.checkbox("Show Entropy Heatmaps"):
                st.subheader("🔥 Channel Entropy Heatmaps")
                
                # Calculate entropy for visualization
                image_array = np.array(image)
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                    image_array = image_array[:, :, :3]
                elif len(image_array.shape) == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                
                selem = disk(radius)
                entropy_red = filters.rank.entropy(img_as_ubyte(image_array[:, :, 0]), selem)
                entropy_green = filters.rank.entropy(img_as_ubyte(image_array[:, :, 1]), selem)
                entropy_blue = filters.rank.entropy(img_as_ubyte(image_array[:, :, 2]), selem)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                im1 = axes[0].imshow(entropy_red, cmap='hot')
                axes[0].set_title('Red Channel Entropy')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].imshow(entropy_green, cmap='hot')
                axes[1].set_title('Green Channel Entropy')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].imshow(entropy_blue, cmap='hot')
                axes[2].set_title('Blue Channel Entropy')
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.info("Please try uploading a different image or adjusting the parameters.")
    
    else:
        st.info("👆 Please upload an image to get started!")
        
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
