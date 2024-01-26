import os
import sys
import time
import argparse
import numpy as np
import pickle
from math import radians
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
import streamlit as st
import tempfile

'''
python tools/convert2fbx.py --input=/home/yusun/BEV_results/video_results.npz --output=/home/yusun/BEV_results/dance.fbx --gender=female
'''
bone_name_from_index = {
    0: 'pelvis', 1: 'left_hip', 2: 'right_hip',
    3: 'spine1', 4: 'left_knee', 5: 'right_knee', 6: 'spine2',
    7: 'left_ankle', 8: 'right_ankle', 9: 'spine3', 10: 'left_foot',
    11: 'right_foot', 12: 'neck', 13: 'left_collar', 14: 'right_collar',
    15: 'head', 16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow',
    19: 'right_elbow',  20: 'left_wrist',  21: 'right_wrist',
    22: 'jaw',
    23: 'left_eye_smplhf', 24: 'right_eye_smplhf', 25: 'left_index1', 26: 'left_index2',
    27: 'left_index3', 28: 'left_middle1', 29: 'left_middle2', 30: 'left_middle3',
    31: 'left_pinky1', 32: 'left_pinky2', 33: 'left_pinky3', 34: 'left_ring1',
    35: 'left_ring2', 36: 'left_ring3', 37: 'left_thumb1', 38: 'left_thumb2',
    39: 'left_thumb3', 40: 'right_index1', 41: 'right_index2', 42: 'right_index3',
    43: 'right_middle1', 44: 'right_middle2', 45: 'right_middle3', 46: 'right_pinky1',
    47: 'right_pinky2', 48: 'right_pinky3', 49: 'right_ring1', 50: 'right_ring2',
    51: 'right_ring3', 52: 'right_thumb1', 53: 'right_thumb2', 54: 'right_thumb3'
}


# To use other avatar for animation, please define the corresponding 3D skeleton like this.
bone_name_from_index_character = {
    0: 'Hips',
    1: 'RightUpLeg',
    2: 'LeftUpLeg',
    3: 'Spine',
    4: 'RightLeg',
    5: 'LeftLeg',
    6: 'Spine1',
    7: 'RightFoot',
    8: 'LeftFoot',
    9: 'Spine2',
    10: 'LeftToeBase',
    11: 'RightToeBase',
    12: 'Neck',
    13: 'LeftHandIndex1',
    14: 'RightHandIndex1',
    15: 'Head',
    16: 'LeftShoulder',
    17: 'RightShoulder',
    18: 'LeftArm',
    19: 'RightArm',
    20: 'LeftForeArm',
    21: 'RightForeArm',
    22: 'LeftHand',
    23: 'RightHand'
}


class PKL2FBX:
    def __init__(self, 
                 input_path, 
                 output_path=os.path.join(os.getcwd(),"result_.fbx"), 
                 fps_source=24, 
                 fps_target=24,
                 gender="neutral",
                 save_fbx = True
    ):
        # super().__init__(*args, **kwargs)
        self.input_path = input_path
        self.save_fbx = save_fbx
        self.output_path = output_path
        self.fps_source = fps_source
        self.fps_target = fps_target
        self.gender = gender

        curr_path = os.getcwd()

        self.male_model_path = os.path.join(
            curr_path, 'model_data/SMPL_unity_v.1.0.0/smplx/smplx-male.fbx')
        self.female_model_path = os.path.join(
            curr_path, 'model_data/SMPL_unity_v.1.0.0/smplx/smplx-female.fbx')
        # self.neutral_model_path = os.path.join(curr_path,'model_data/SMPL_unity_v.1.0.0/smplx/smplx-neutral.fbx')
        self.neutral_model_path = os.path.join(
            curr_path, 'model_data/SMPL_unity_v.1.0.0/smplx/smplx-neutral-basic.fbx')
        self.character_model_path = None

        self.support_formats = ['.fbx', '.glb', '.bvh']
        self.bone_name_from_index = bone_name_from_index

        self.adding_pose_text = "Adding pose operation is in progress. Please wait."
        self.adding_pose_bar = st.progress(0, text=self.adding_pose_text)
        self.adding_pose_bar_value = 0

        # self.fbx_data = self.call()

    def Rodrigues(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])
        return (cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # Setup scene
    def setup_scene(self, model_path, fps_target):
        scene = bpy.data.scenes['Scene']

        ###########################
        # Engine independent setup
        ###########################

        scene.render.fps = fps_target

        # Remove default cube
        if 'Cube' in bpy.data.objects:
            bpy.data.objects['Cube'].select_set(True)
            bpy.ops.object.delete()

        # Import gender specific .fbx template file
        bpy.ops.import_scene.fbx(filepath=model_path)

    # Process single pose into keyframed bone orientations
    def process_pose(self, current_frame, pose, trans, pelvis_position, gender):

        if pose.shape[0] == 72:
            rod_rots = pose.reshape(24, 3)
        elif pose.shape[0] == 165:
            rod_rots = pose.reshape(55, 3)
        else:
            rod_rots = pose.reshape(26, 3)

        mat_rots = [self.Rodrigues(rod_rot) for rod_rot in rod_rots]

        # Set the location of the Pelvis bone to the translation parameter
        if gender == 'female':
            armature = bpy.data.objects['SMPLX-female']
        elif gender == 'male':
            armature = bpy.data.objects['SMPLX-male']
        elif gender == 'neutral':
            armature = bpy.data.objects['SMPLX-neutral']

        # armature = bpy.data.objects['SMPLX-female']
        bones = armature.pose.bones

        root_location = Vector(
            (trans[1], trans[2], trans[0]))

        # Set absolute pelvis location relative to Pelvis bone head
        bones[self.bone_name_from_index[0]].location = root_location

        # bones['Root'].location = Vector(trans)
        bones[self.bone_name_from_index[0]].keyframe_insert(
            'location', frame=current_frame)

        for index, mat_rot in enumerate(mat_rots, 0):
            # if index >= 24:
            #     continue

            if index >= 55:
                continue

            bone = bones[self.bone_name_from_index[index]]

            bone_rotation = Matrix(mat_rot).to_quaternion()
            quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))
            # quat_x_n135_cw = Quaternion((1.0, 0.0, 0.0), radians(-135))
            # quat_x_p45_cw = Quaternion((1.0, 0.0, 0.0), radians(45))
            # quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))
            quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))

            if index == 0:
                # Rotate pelvis so that avatar stands upright and looks along negative Y avis
                bone.rotation_quaternion = (
                    quat_x_90_cw @ quat_z_90_cw) @ bone_rotation
            else:
                bone.rotation_quaternion = bone_rotation

            bone.keyframe_insert('rotation_quaternion', frame=current_frame)

        return

    def EMA(self, arr, smoothing_factor):
        i = 1
        moving_averages = []
        moving_averages.append(arr[0])
        while i < len(arr):
            window_average = smoothing_factor * \
                arr[i] + (1 - smoothing_factor) * moving_averages[-1]
            moving_averages.append(window_average)
            i += 1
        return moving_averages

    def process_poses(self,
                      input_path,
                      gender,
                      fps_source,
                      fps_target,
                      subject_id=-1):

        print('Processing: ' + input_path)

        with open(input_path, "rb") as f:
            frame_results = pickle.load(f)

        sequence_results = []

        poses, trans = [], []
        poses, trans = np.zeros((len(frame_results['pred_thetas']), 165)
                                ), np.zeros((len(frame_results['transl']), 3))
        for inds in range(len(frame_results['pred_thetas'])):
            # poses[inds] = frame_results['pred_thetas'][inds][:72]
            poses[inds] = frame_results['pred_thetas'][inds]
            trans[inds] = frame_results['transl'][inds]

        trans = self.EMA(trans, 0.5)

        if gender == 'female':
            model_path = self.female_model_path
        elif gender == 'male':
            model_path = self.male_model_path

        elif gender == 'neutral':
            model_path = self.neutral_model_path
        elif gender == 'character':
            model_path = self.character_model_path
        else:
            print('ERROR: Unsupported gender: ' + gender)
            sys.exit(1)

        # Limit target fps to source fps
        if fps_target > fps_source:
            fps_target = fps_source

        print('Gender:', gender)
        print('Number of source poses: ', poses.shape[0])
        print('Source frames-per-second: ', fps_source)
        print('Target frames-per-second: ', fps_target)
        print('--------------------------------------------------')

        self.setup_scene(model_path, fps_target)

        scene = bpy.data.scenes['Scene']
        sample_rate = int(fps_source/fps_target)
        scene.frame_end = (int)(poses.shape[0]/sample_rate)
        # print("from convert_pkl2fbx.py: ", scene.frame_end)

        # Retrieve pelvis world position.
        # Unit is [cm] due to Armature scaling.
        # Need to make copy since reference will change when bone location is modified.
        armaturee = bpy.data.armatures[0]

        if gender == 'female':
            ob = bpy.data.objects['SMPLX-female']
        elif gender == 'male':
            ob = bpy.data.objects['SMPLX-male']
        elif gender == 'neutral':
            ob = bpy.data.objects['SMPLX-neutral']

        armature = ob.data

        bpy.ops.object.mode_set(mode='EDIT')
        # get specific bone name 'Bone'
        pelvis_bone = armature.edit_bones[self.bone_name_from_index[0]]
        # pelvis_bone = armature.edit_bones['f_avg_Pelvis']
        pelvis_position = Vector(pelvis_bone.head)
        bpy.ops.object.mode_set(mode='OBJECT')

        source_index = 0
        frame = 1

        offset = np.array([0.0, 0.0, 0.0])

        # for percent_complete in range(100):
        #     time.sleep(0.01)
        #     my_bar.progress(percent_complete + 1, text=progress_text)

        while source_index < poses.shape[0]:
            print('Adding pose: ' + str(source_index))

            # Go to new frame
            scene.frame_set(frame)

            self.process_pose(frame, poses[source_index],
                              (trans[source_index] - offset), pelvis_position, gender=gender)
            source_index += sample_rate
            frame += 1
            # print(source_index)
            # print(poses.shape[0])
            # print(int((source_index/poses.shape[0])*100))
            self.adding_pose_bar_value = int((source_index/poses.shape[0])*97)
            self.adding_pose_bar.progress(
                self.adding_pose_bar_value, text=self.adding_pose_text)

        return frame

    def rotate_armature(self, use, gender):
        if use == True:
            # Switch to Pose Mode
            bpy.ops.object.posemode_toggle()

            # Find the Armature & Bones
            if gender == 'female':
                ob = bpy.data.objects['SMPLX-female']
            elif gender == 'male':
                ob = bpy.data.objects['SMPLX-male']
            elif gender == 'neutral':
                ob = bpy.data.objects['SMPLX-neutral']

            # ob = bpy.data.objects['SMPLX-female']
            armature = ob.data
            bones = armature.bones
            rootbone = bones[0]

            # Find the Root bone
            for bone in bones:
                if "avg_root" in bone.name:
                    rootbone = bone

            rootbone.select = True

            # Rotate the Root bone by 90 euler degrees on the Y axis. Set --rotate_Y=False if the rotation is not needed.
            bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(
                False, True, False), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

            # Rotate the Root bone by 90 euler degrees on the Y axis. Set --rotate_Y=False if the rotation is not needed.
            bpy.ops.transform.rotate(value=1.5708, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(
                False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)

            # Revert back to Object Mode
            bpy.ops.object.posemode_toggle()

    def export_animated_mesh(self, output_path, gender):
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Fix Rotation
        # rotate_armature(args.rotate_y, gender=gender)
        self.rotate_armature(True, gender=gender)

        # Select only skinned mesh and rig
        bpy.ops.object.select_all(action='DESELECT')
        if gender == 'female':
            bpy.data.objects['SMPLX-female'].select_set(True)
            bpy.data.objects['SMPLX-female'].children[0].select_set(True)
        elif gender == 'male':
            bpy.data.objects['SMPLX-male'].select_set(True)
            bpy.data.objects['SMPLX-male'].children[0].select_set(True)
        elif gender == 'neutral':
            bpy.data.objects['SMPLX-neutral'].select_set(True)
            bpy.data.objects['SMPLX-neutral'].children[0].select_set(True)

        # bpy.data.objects['SMPLX-female'].select_set(True)
        # bpy.data.objects['SMPLX-female'].children[0].select_set(True)

        if output_path.endswith('.glb'):
            print('Exporting to glTF binary (.glb)')
            # Currently exporting without shape/pose shapes for smaller file sizes
            bpy.ops.export_scene.gltf(
                filepath=output_path, export_format='GLB', export_selected=True, export_morph=False)
        elif output_path.endswith('.fbx'):
            print('Exporting to FBX binary (.fbx)')

            if self.save_fbx:
                # No use tempfile
                finish = bpy.ops.export_scene.fbx(filepath=output_path,
                                                use_selection=True, add_leaf_bones=False)
                print(list(finish)[0])
                if list(finish)[0] == 'FINISHED':
                    self.adding_pose_bar_value = 100
                    self.adding_pose_bar.progress(
                        self.adding_pose_bar_value, text=self.adding_pose_text)
                
                
                with open(output_path, "rb") as f:  # Open a file in binary write mode
                    f.seek(0)
                    fbx_data = f.read()
                    return fbx_data
            elif self.save_fbx == False:
                # use tempfile
                with tempfile.TemporaryFile() as temp_f:
                    bpy.ops.export_scene.fbx(filepath=f"{temp_f.name}.fbx", use_selection=True, add_leaf_bones=False)  # Export to temporary file
                    
                    self.adding_pose_bar_value = 100
                    self.adding_pose_bar.progress(
                        self.adding_pose_bar_value, text=self.adding_pose_text)
                    
                    with open(f"{temp_f.name}.fbx", "rb") as f:  # Open a file in binary write mode
                        f.seek(0)
                        fbx_data = f.read()

                    os.remove(f"{temp_f.name}.fbx")

                    return fbx_data
                    
                
        elif output_path.endswith('.bvh'):
            bpy.ops.export_anim.bvh(filepath=output_path,
                                    root_transform_only=False)
        else:
            print('ERROR: Unsupported export format: ' + output_path)
            sys.exit(1)

        return

    def run(self):
        startTime = time.perf_counter()
        cwd = os.getcwd()
        # Turn relative input/output paths into absolute paths
        if not self.input_path.startswith(os.path.sep):
            self.input_path = os.path.join(cwd, self.input_path)
        if not self.output_path.startswith(os.path.sep):
            self.output_path = os.path.join(cwd, self.output_path)

        if os.path.splitext(self.output_path)[1] not in self.support_formats:
            print('ERROR: Invalid output format, we only support',
                  self.support_formats)
            sys.exit(1)

        # Process pose file
        poses_processed = self.process_poses(
            input_path=self.input_path,
            gender=self.gender,
            fps_source=self.fps_source,
            fps_target=self.fps_target,
            # subject_id=args.subject_id
        )

        # print(f"from convert_pkl2fbx.py: {poses_processed}")

        fbx_data =self.export_animated_mesh(self.output_path, gender=self.gender)

        print('--------------------------------------------------')
        print('Animation export finished, save to ', self.output_path)
        print('Poses processed: ', poses_processed)
        print('Processing time : ', time.perf_counter() - startTime)
        print('--------------------------------------------------')

        return fbx_data

    def call(self):
        if bpy.app.background:

            # delete all objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            # Resetting to Factory Settings
            bpy.ops.wm.read_factory_settings(use_empty=True)

            fbx_data = self.run()
            # print(fbx_data)
            return fbx_data
    def __str__(self):
        fbx_data = self.call()
        if self.save_fbx == True:
            return self.output_path
        else:
            return str(fbx_data)           
    
if __name__ == '__main__':
    fbx1 = PKL2FBX('Pkls/final_all_3.pkl',save_fbx=False)
    print(f"fbx1: {fbx1}")

    fbx2 = PKL2FBX('Pkls/final_all_4.pkl')
    print(f"fbx2: {fbx2}")

    fbx3 = PKL2FBX('pkl_file.pkl')
    print(f"fbx3: {fbx3}")

    
