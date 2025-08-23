import streamlit as st
from base_model import BaseModel
from mongodb import database_name, collection_name

try:
    if st.button("Get all student names", key="get_student_names_button"):
        model = BaseModel(collection_name)
        student_names = model.get_all_student_names()
        st.write("Students : ", ", ".join(student_names))

    if st.button("Get all student details", key="get_student_details_button"):
        model = BaseModel(collection_name)
        student_details = model.get_all_student_details()
        st.write("Student Details : ", student_details)

    if st.button("Add new student details", key="add_student_details_button"):
        model = BaseModel(collection_name)
        student_name = st.text_input("Enter student name", key="new_student_name_input")
        marks = st.number_input("Enter student marks", key="new_student_marks_input")
        age = st.number_input("Enter student age", key="new_student_age_input")

        if st.button("Add Student", key="add_student_button"):
            model.add_student(student_name, marks, age)
            st.success(f"Student {student_name} added successfully.")

    student_name = st.text_input("Enter student name to update marks", key="student_name_input")
    if student_name:
        marks = st.number_input("Enter new marks", key="marks_input")
        if marks:
            model = BaseModel(collection_name)
            student_details = model.update_students_marks(student_name, marks)

    student_name = st.text_input("Enter student name to delete from database", key="student_name_input_to_delete")
    if st.button("Delete Student", key="delete_student_button"):
        model = BaseModel(collection_name)
        model.delete_student(student_name)
        st.write(f"Student {student_name} deleted successfully.")

except Exception as e:
    st.toast(f"Error: {e}")