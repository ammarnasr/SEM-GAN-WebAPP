import streamlit as st
from Bird_Image_Generator import GenerateImages
from PIL import Image

#Configuring the page
st.set_page_config(page_title="Bird Image Generator", page_icon="🐦", layout="wide")

#Title
st.title("Bird Image Generator")

#Description
with st.expander("ℹ️ Description"):
    st.write("TBD: Add a description of the project.")


#Input
st.markdown("""
**Note:** The model is trained on the CUB-200-2011 dataset. So, the captions should be related to birds.
""")
#Input
example_caption = "This is a bird with a yellow belly, black head and breast and a black wing."
Input_Captions = st.text_area("Enter the captions for the images you want to generate. write each caption in a separate line.", example_caption, height=150)
Input_Captions = [Input_Caption.strip() for Input_Caption in Input_Captions.split("\n")]
st.write("You have entered", len(Input_Captions), "captions. Click on the button below to generate the images.")


#Generate Button
if st.button("Generate"):
    with st.spinner("Generating Images..."):
        image_paths = GenerateImages(Input_Captions)
    st.success("Images Generated Successfully!")
    caps_counter = 0
    for i, image_path in enumerate(image_paths):
        if not image_path.endswith("2.png"):
            continue
        image = Image.open(image_path)
        st.image(image, caption=Input_Captions[caps_counter], use_column_width=False)
        caps_counter += 1


        
#Footer
st.markdown("---")
st.markdown("""Made with ❤️ by [Ammar Khairi]""")

#Sidebar
st.sidebar.markdown("### About")
# Link to Github repo of the project:https://github.com/ammarnasr/SEM-GAN-WebAPP.git
st.sidebar.markdown("Add a link to the Github repo of the project and Arxiv paper.")