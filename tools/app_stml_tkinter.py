import streamlit as st
from convert_pkl2fbx import ConvertPkl2Fbx
# import pickle
# from io import StringIO
import bpy
import time
import os
import tkinter as tk
from tkinter import filedialog

def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

def convert_UI():
    # st.title('FTECH AI - Convert PKL to FBX ')
    # st.title('from SVT demo')

    title = '<p style="font-size: 30px; text-align: center; color: white; font-weight: bold;">FTECH AI - Convert PKL to FBX from SVT demo</p>'
    st.markdown(title, unsafe_allow_html=True)


    pkl_file = st.file_uploader("Input your pkl file here:")
    #pkl folder
    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Select Folder")

    if folder_select_button:
        selected_folder_path = select_folder()
        st.session_state.folder_path = selected_folder_path
        print(selected_folder_path)
        print(os.listdir(selected_folder_path))

    input_path = "pkl_file.pkl"
    output_path = 'result.fbx'
    fps_source = 24
    fps_target = 24
    gender = 'neutral'

    gender = st.selectbox(
        'Please choose a gender: ',
        ('neutral', 'female', 'male'))

    if st.button("Convert"):
        print("selected_folder_path: ", st.session_state.folder_path)
        if pkl_file is not None:

            # delete pkl and fbx files if exist
            if os.path.exists("pkl_file.pkl"):
                os.remove("pkl_file.pkl")
            if os.path.exists("result.fbx"):
                os.remove("result.fbx")

            bytes_data = pkl_file.getvalue()

            with open("pkl_file.pkl", "wb") as f:
                f.write(bytes_data)

            
            convert_pkl2fbx = ConvertPkl2Fbx()
            convert_pkl2fbx.convert(input_path, output_path,fps_source, fps_target, gender)

                # my_bar.progress(percent_complete + 1, text=progress_text)

            st.download_button(
                label="Download Generated FBX File",
                # Open the temporary file in binary mode
                data=open("result.fbx", "rb"),
                file_name="generated_fbx.fbx"  # Customize the filename for download
            )
        elif st.session_state.folder_path is not None:
            print("converting files in the folder...")
            convert_pkl2fbx = ConvertPkl2Fbx()
            output_path = "output_fbxs"
            convert_pkl2fbx.convert(st.session_state.folder_path, output_path,fps_source, fps_target, gender)

            st.download_button(
                label="Download Generated FBX File",
                # Open the temporary file in binary mode
                data=open(f"{output_path}/results.zip", "rb"),
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