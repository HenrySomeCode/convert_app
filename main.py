from tools.convert_FBX import PKL2FBX

if __name__ == '__main__':
    fbx1 = PKL2FBX('Pkls/final_all_3.pkl',save_fbx=False)
    print(f"fbx1: {fbx1}")

    fbx2 = PKL2FBX('Pkls/final_all_4.pkl')
    print(f"fbx2: {fbx2}")

    fbx3 = PKL2FBX('pkl_file.pkl')
    print(f"fbx3: {fbx3}")