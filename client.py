import streamlit as st
import requests
from config import HOST
from ultralytics import YOLO
import re
from time import time
import os
import shutil

# Настройка заголовка приложения
st.title("Путешествие в мир ИИ")

# Инициализация состояния сессии
if 'team_registered' not in st.session_state:
    st.session_state.team_registered = False
    st.session_state.team_name = ""
    st.session_state.participants_list = []

# Создание вкладок
tab1, tab2, tab3 = st.tabs(["Регистрация команды", "Ввод данных о животных", "Обучение YOLO"])

# Первая вкладка: Регистрация команды
with tab1:
    st.header("Форма регистрации команды")
    
    
    team_name = st.text_input("Название команды:")
    participant_names = st.text_area("Имена участников (через запятую):")
    
    if st.button("Зарегистрировать команду"):
        if team_name and participant_names:
            st.session_state.team_registered = True
            st.session_state.team_name = team_name
            st.session_state.participants = participant_names

            data = {
                "name": team_name,
                "participants": participant_names
            }
            response = requests.post(f"{HOST}/register_team", json=data, verify=False)
            response = response.json()

            if response["status"] == "ok":
                st.success(f"Команда '{team_name}' зарегистрирована с участниками: {st.session_state.participants}")
            else: 
                st.error(response["message"])
                st.session_state.team_registered = False
                st.session_state.team_name = ""
                st.session_state.participants = ""
        else:
            st.error("Пожалуйста, заполните все поля.")

# Вторая вкладка: Ввод данных о зверях
with tab2:
    teams = [i["name"] for i in requests.get(f"{HOST}/get_teams", verify=False).json()["teams"]]
    classes = requests.get(f"{HOST}/get_classes", verify=False).json()["classes"]

    idx = teams.index(st.session_state.team_name) if st.session_state.team_name != "" else None
    team = st.selectbox( "Название команды", teams, index=idx, placeholder="Выберите команду", key='team_name_2')
    

    st.header("Форма ввода данных о подсчёте объектов")
    class_counts = {
        "team": team,
    }

    for i in classes:
        count = st.number_input(f"Количество объектов класса {i['name'].lower()} на видео:", min_value=0, step=1)
        class_counts[i["name"]] = count

    if st.button("Отправить данные о зверях"):
        if team is None:
            st.error("Выбор команды является обязательным!")
        else:
            response = requests.post(f"{HOST}/save_class_counts", json=class_counts, verify=False)
            response = response.json()
            if response["status"] == "ok":
                st.success("Данные успешно отправлены!")
            else: 
                st.error(response["message"])

with tab3:
    teams = [i["name"] for i in requests.get(f"{HOST}/get_teams", verify=False).json()["teams"]]
    classes = requests.get(f"{HOST}/get_classes", verify=False).json()["classes"]

    idx = teams.index(st.session_state.team_name) if st.session_state.team_name != "" else None
    team = st.selectbox("Название команды", teams, index=idx, placeholder="Выберите команду", key="team_name_3")

    st.header("Настройки обучения модели YOLO")
    epochs = st.number_input("Количество эпох:", min_value=1, value=5)
    imgsz = st.number_input("Размер изображений:", min_value=32, value=128, max_value=400)
    batch_size = st.number_input("Размер батча:", min_value=4, value=16)
    learning_rate = st.number_input("Коэффициент скорости обучения:", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%0.4f")

    best_mAP50 = 0
    best_epoch = 0
    mAP50_values = []

    if st.button("Обучить модель"):
        if os.path.exists("runs"):
            shutil.rmtree("runs")

        best_mAP50 = 0
        best_epoch = 0
        mAP50_values = []
        
        st.success(f"Обучение модели YOLO начато с параметрами:\n- Эпохи: {epochs}\n- Размер изображений: {imgsz}\n- Размер батча: {batch_size}\n- Скорость обучения: {round(learning_rate, 4)}")
        
        model = YOLO('yolov8n.pt')
        
        # Цикл по эпохам
        for epoch in range(epochs):
            st.write(f"Начало обучения эпохи {epoch + 1}/{epochs}...")
            start_time = time()
            
            with st.spinner(f'Обучение эпохи {epoch + 1}/{epochs}...'):
                results = model.train(data=f"{os.getcwd()}/data/data.yaml", 
                                    epochs=1,  # Обучаем только одну эпоху за раз
                                    imgsz=imgsz,
                                    batch=batch_size,
                                    lr0=round(learning_rate, 4),
                                    project= f'{os.getcwd()}/runs', # ПАПКА
                                    name='yolo_training_results')
            
            # Извлечение нужных данных из results
            results_str = str(results)
            
            # Извлечение значения metrics/mAP50(B)
            pattern = r"'metrics/mAP50\(B\)': ([0-9\.]+)"
            mAP50_value = re.search(pattern, results_str).group(1)
            mAP50_values.append(mAP50_value)

            # Отображение результатов обучения
            st.write(f"Результаты обучения на эпохе {epoch + 1}:")
            st.info(f"Результаты обучения на эпохе {epoch + 1}:\n- metrics/mAP50(B): {mAP50_value}\n- Время обучения в секундах: {round(time()-start_time)}")
        best_map = max(mAP50_values)
        best_map_idx = mAP50_values.index(best_map)
        params = {
            "epochs": epochs,
            "imgsz": imgsz,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
        data = {
            "name": team,
            "score": best_map,
            "params": str(params)
        }
        response = requests.post(f"{HOST}/commit_yolo_results", json=data, verify=False)
        response = response.json()
        if response["status"] == "ok":
            st.success(f"Обучение модели YOLO завершено! Лучший mAP: {best_map}")
        else: 
            st.error(response["message"])
    
        st.subheader("Предсказанное видео")
        best_folder = os.listdir(f"{os.getcwd()}/runs")[best_map_idx]
        print(best_folder)
        model = YOLO(f'./runs/{best_folder}/weights/best.pt')
        # Открытие видео и получение предсказаний
        input_video_path = './data/test_video.mp4'
        yolo_video_path = './runs/predict/test_video.avi'
        output_video_path = './runs/predict/test_video.mp4'
        model.predict(input_video_path, save=True, project=f'{os.getcwd()}/runs')
        os.system(f"ffmpeg -i {yolo_video_path} -vcodec libx264 {output_video_path}")
        # отрисовка
        video_file = open(output_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
