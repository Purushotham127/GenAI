from fastapi import APIRouter, Request
import os
import sys
import requests
current_dir = os.path.dirname(os.path.abspath(__file__))
three_dirs_up = os.path.abspath(os.path.join(current_dir, "../.."))
one_dir_up = os.path.abspath(os.path.join(current_dir, "../"))
env_path = os.path.join(one_dir_up, '.env')
sys.path.insert(0, three_dirs_up)

from Mongodb_Class.base_model import BaseModel, Student
import dotenv

dotenv.load_dotenv(env_path)
collection_name = os.getenv("COLLECTION_NAME")

router = APIRouter()

@router.get("/get_studentnames")
def get_student_names():
    try:
        model = BaseModel(collection_name)
        return model.get_all_student_names()
    except Exception as e:
        return {"error": str(e)}

@router.get("/get_student_details/{student_name}")
def get_student_details(student_name: str):
    try:
        model = BaseModel(collection_name)
        return model.get_student_details(student_name)
    except Exception as e:
        return {"error": str(e)}

@router.post("/add_student")
async def add_student(request: Request):
    try:
        student_details = await request.json()
        model = BaseModel(collection_name)
        model.add_student(student_details.name, student_details.marks, student_details.age)
        return {"message": "Student added successfully"}
    except Exception as e:
        return {"error": str(e)}

@router.put("/update_student_marks/student_name={student_name},marks={marks}")
def update_student_marks(student_name: str, marks: float):
    try:
        model = BaseModel(collection_name)
        model.update_students_marks(student_name, marks)
    except Exception as e:
        return {"error": str(e)}
    return {"message": "Student marks updated successfully"}

@router.delete("/delete_student/{student_name}")
def delete_student(student_name: str):
    try:
        model = BaseModel(collection_name)
        model.delete_student(student_name)
        return {"message": "Student deleted successfully"}
    except Exception as e:
        return {"error": str(e)}

@router.get("/get_all_student_details")
def get_all_student_details():
    model = BaseModel(collection_name)
    return model.get_all_student_details()