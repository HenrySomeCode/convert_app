import streamlit as st
from convert_pkl2fbx import ConvertPkl2Fbx
import os
import shutil
import zipfile
def convert_UI():
    # st.title('FTECH AI - Convert PKL to FBX ')
    # st.title('from SVT demo')

    title = '<p style="font-size: 30px; text-align: center; color: white; font-weight: bold;">FTECH AI - Convert PKL to FBX from SVT demo</p>'
    st.markdown(title, unsafe_allow_html=True)


    pkl_files = st.file_uploader("Input your pkl file here:", accept_multiple_files=True)

    # if pkl_file is not None:
    #     print(pkl_file.name)

    input_path = "pkl_file.pkl"
    output_path = 'result.fbx'
    fps_source = 24
    fps_target = 24
    gender = 'neutral'

    gender = st.selectbox(
        'Please choose a gender: ',
        ('neutral', 'female', 'male'))
    is_a_zip_file = False

    if st.button("Convert"):
        if pkl_files is not None:
            fbx_files = []
            for pkl_file in pkl_files:
                # a single pkl file 
                if(pkl_file.name.endswith(("pkl","pk"))):
                    # delete pkl and fbx files if exist
                    if os.path.exists("pkl_file.pkl"):
                        os.remove("pkl_file.pkl")
                    if os.path.exists("result.fbx"):
                        os.remove("result.fbx")

                    bytes_data = pkl_file.getvalue()

                    with open("pkl_file.pkl", "wb") as f:
                        f.write(bytes_data)

                    convert_pkl2fbx = ConvertPkl2Fbx()
                    output_path = pkl_file.name.replace("pkl","fbx")
                    fbx_files.append(output_path)
                    convert_pkl2fbx.convert(input_path, output_path,fps_source, fps_target, gender)

                        # my_bar.progress(percent_complete + 1, text=progress_text)

                    # st.download_button(
                    #     label="Download Generated FBX File",
                    #     # Open the temporary file in binary mode
                    #     data=open("result.fbx", "rb"),
                    #     file_name=pkl_file.name.replace("pkl","fbx")  # Customize the filename for download
                    # )
                elif(pkl_file.name.endswith("zip")):
                    print(pkl_file.name)

                    is_a_zip_file = True

                    convert_pkl2fbx = ConvertPkl2Fbx()
                    if not os.path.exists("tempDir"):
                        os.makedirs("tempDir")

                    with open(os.path.join("tempDir",pkl_file.name),"wb") as f:
                        f.write(pkl_file.getbuffer())
                    
                    convert_pkl2fbx.convert(os.path.join("tempDir",pkl_file.name), "output_fbxs",fps_source, fps_target, gender)

                    shutil.rmtree("tempDir")

                    fbx_files.append("output_fbxs/results.zip")


                        # my_bar.progress(percent_complete + 1, text=progress_text)

                    st.download_button(
                        label="Download Generated FBX File",
                        # Open the temporary file in binary mode
                        data=open("output_fbxs/results.zip", "rb"),
                        file_name="results.zip"  # Customize the filename for download
                    )
            if len(pkl_files) == 1 and not is_a_zip_file:
                st.download_button(
                    label="Download Generated FBX File",
                    # Open the temporary file in binary mode
                    data=open(fbx_files[0], "rb"),
                    file_name=pkl_file.name.replace("pkl","fbx")  # Customize the filename for download
                )
            elif len(pkl_files) > 1:                
                zip_f_path = "output_fbxs/results.zip"

                if os.path.exists(zip_f_path):
                    os.remove(zip_f_path) 

                with zipfile.ZipFile(zip_f_path, "w") as zipf:
                    for file in fbx_files:
                        zipf.write(file)

                st.download_button(
                    label="Download Generated FBX File",
                    # Open the temporary file in binary mode
                    data=open(zip_f_path, "rb"),
                    file_name="results.zip"  # Customize the filename for download
                )


    # Add a logout button
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()  # Rerun the app to display the login UI

def login_UI():
    title = '<p style="font-size: 30px; text-align: center; color: white; font-weight: bold;">Login</p>'
    st.markdown(title, unsafe_allow_html=True)
    # st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in credentials and credentials[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()  # Rerun the app to display the new UI

        else:
            st.error("Invalid credentials")

# Define login credentials 
credentials = {"timi_svt": "ftech", "user2": "password2"}

# if os.environ.get('DISPLAY', '') == '':
#     print('No display found. Using Xming server at localhost:0')
#     os.environ['DISPLAY'] = 'localhost:0'

# Create a session state to track login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login page
if not st.session_state.logged_in:
    login_UI()
if st.session_state.logged_in:
    convert_UI()