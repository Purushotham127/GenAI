import streamlit as st
import requests
import dotenv
import os
from Mongodb_Class.base_model import Student
dotenv.load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

try:
    if st.button("Get all student names", key="get_student_names_button"):
        response = requests.get(f"{BACKEND_URL}/get_studentnames")
        if response.status_code == 200:
            student_names = response.json()
            st.write("Students : ", ", ".join(student_names))
        else:
            st.error("Failed to retrieve student names.")

    if st.button("Get all student details", key="get_student_details_button"):
        response = requests.get(f"{BACKEND_URL}/get_all_student_details")
        if response.status_code == 200:
            student_details = response.json()
            st.write("All Student Details : ", student_details)
        else:
            st.error("Failed to retrieve all student details.")

    if st.button("Add new student details", key="add_student_details_button"):
        student_name = st.text_input("Enter student name", key="new_student_name_input")
        marks = st.number_input("Enter student marks", key="new_student_marks_input")
        age = st.number_input("Enter student age", key="new_student_age_input")
        student_details = Student(name=student_name, marks=marks, age=age)

        if age and marks and student_name:
            response = requests.post(f"{BACKEND_URL}/add_student", json=student_details.dict())
            if response.status_code == 201:
                st.success(f"Student {student_name} details added successfully. Status code 201.")
            elif response.status_code == 200:
                st.success(f"Student {student_name} details updated successfully. Status code 200.")
            else:
                st.error(f"Failed to add student details, reason: {response.text}")

    student_name = st.text_input("Enter student name to update marks", key="student_name_input")
    if student_name:
        marks = st.number_input("Enter new marks", key="marks_input")
        if marks:
            response = requests.put(f"{BACKEND_URL}/update_student_marks/student_name={student_name},marks={marks}")
            if response.status_code == 200:
                st.success(f"Updated marks for {student_name}.")
            else:
                st.error("Failed to update marks.")

    student_name = st.text_input("Enter student name to delete from database", key="student_name_input_to_delete")
    if student_name and st.button("Delete Student", key="delete_student_button"):
        response = requests.delete(f"{BACKEND_URL}/delete_student/{student_name}")
        if response.status_code == 200:
            st.success(f"Deleted student {student_name}.")
        else:
            st.error("Failed to delete student.")

except Exception as e:
    st.toast(f"Error: {e}")