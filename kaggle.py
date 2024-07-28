import subprocess
import time
import pandas as pd


class Kaggle:
    """
    Взаимодействие с платформой Kaggle.
    """

    __COLUMNS = ['fileName', 'date', 'description', 'status', 'publicScore', 'privateScore']

    def __init__(self, competition: str, verbose: int = 0):
        """
        Инициализация взаимодействия с платформой Kaggle.

        Аргументы:
        - competition: название соревнования.
        - verbose: режим верболизации: 0 (тихий) или 1 (вывод сообщений)."""

        self.__competition = competition
        self.__verbose = verbose

    def download_data_files(self, file_names: [str], dir_path: str):
        """
        Скачивание файлов исходных данных.

        Аргументы:
        - files: список названий скачиваемых файлов.
        - dir_path: путь к папке скачанных файлов.
        """

        downloads_cnt = len(file_names)
        successful_downloads_cnt = 0
        failed_downloads_cnt = 0
        for index, file_name in enumerate(file_names):
            cmd = f'kaggle competitions download -c {self.__competition} -f "{file_name}" -p "{dir_path}" -q'
            if self.__verbose:
                print(f'{index + 1}/{downloads_cnt}: Downloading data file "{file_name}" '
                      f'of competition "{self.__competition}"...')
            popen = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            while True:
                time.sleep(1)
                popen.poll()
                if popen.returncode is None:
                    continue
                if popen.returncode == 0:
                    successful_downloads_cnt += 1
                    if self.__verbose:
                        print(f'{index + 1}/{downloads_cnt}: Successfully downloaded.')
                else:
                    failed_downloads_cnt += 1
                    if self.__verbose:
                        print(f'{index + 1}/{downloads_cnt}: Not downloaded.')
                break
        if self.__verbose:
            result_str = f'{successful_downloads_cnt} out of {downloads_cnt} ' \
                         'data files have been successfully downloaded.'
            print('-' * len(result_str))
            print(result_str, end='\n\n')
        return successful_downloads_cnt

    def send_submission_files(self, file_paths: [str], descriptions: [str]):
        """Отправка файлов решений на проверку.

        Аргументы:
        - descriptions: список описаний решений.
        - file_paths: список путей к файлам решений."""
        submissions_cnt = min(len(file_paths), len(descriptions))
        successful_submissions_cnt = 0
        failed_submissions_cnt = 0
        for index, (file_path, description) in enumerate(zip(file_paths, descriptions)):
            cmd = f'kaggle competitions submit -c {self.__competition} -f "{file_path}" -m "{description}" -q'
            if self.__verbose:
                print(f'{index + 1}/{submissions_cnt}: Sending file "{file_path}" '
                      f'of submission "{description}" to competition "{self.__competition}"...')
            popen = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            while True:
                time.sleep(1)
                popen.poll()
                if popen.returncode is None:
                    continue
                if popen.returncode == 0:
                    successful_submissions_cnt += 1
                    if self.__verbose:
                        print(f'{index + 1}/{submissions_cnt}: Successfully submitted.')
                else:
                    failed_submissions_cnt += 1
                    if self.__verbose:
                        print(f'{index + 1}/{submissions_cnt}: Not submitted.')
                break
        if self.__verbose:
            result_str = f'{successful_submissions_cnt} out of {submissions_cnt} ' \
                         'submission files have been successfully sent.'
            print('-' * len(result_str))
            print(result_str, end='\n\n')
        return successful_submissions_cnt

    def receive_submission_scores(self, descriptions: [str]) -> pd.DataFrame | None:
        """
        Прием результатов проверки решений.

        Аргументы:
        - descriptions: список описаний решений.
        """

        trimmed_descriptions = [description.replace('"', '') for description in descriptions]

        # Запрашиваем список результатов
        if self.__verbose:
            print('Receiving data from Kaggle...')
        cmd = f'kaggle competitions submissions -c {self.__competition}'

        while True:
            popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1)
            popen.poll()
            if isinstance(popen.returncode, int):
                if self.__verbose:
                    print(f'Can\'t retrieve data from Kaggle. Error: {popen.stdout.read().decode("ansi")}')
                return
            lines = popen.stdout.read().decode('ansi')
            if lines.find(' pending ') >= 0:
                continue
            if self.__verbose:
                print('Data received. Waiting for submissions pending complete...')
                break

        # Находим строку заголовков
        lines = lines.split('\n')

        for index, line in enumerate(lines):
            if line[:-1].split() == self.__COLUMNS:
                break
        if index == len(lines) - 1:
            if self.__verbose:
                print(f'Can\'t retrieve data from Kaggle. Error: "{lines}"')
            return

        if self.__verbose:
            print('Data received.')

        # Находим положение колонок в тексте
        header_start_positions = [line.find(column) for column in self.__COLUMNS]
        header_end_positions = header_start_positions[1:]
        header_end_positions.append(len(line))

        # Оставляем строки с результатами
        lines = lines[index + 2:]

        # Извлекаем данные из строк результатов
        data = [
            [
                line[header_start_position: header_end_position].strip()
                for header_start_position, header_end_position in zip(header_start_positions, header_end_positions)
            ] for line in lines
        ]

        # Создаем датасет из полученных результатов
        result = pd.DataFrame(data, columns=self.__COLUMNS)
        result['publicScore'] = pd.to_numeric(result['publicScore'], errors='coerce')
        result['privateScore'] = pd.to_numeric(result['privateScore'], errors='coerce')
        result['date'] = pd.to_datetime(result['date'])

        # Оставляем только результаты отправленных предсказаний
        result = result[result['description'].isin(trimmed_descriptions)]

        # Т.к. могут быть одноименные файлы, то берем самые поздние
        indexes = sorted([indexes[0] for indexes in result.groupby('description').groups.values()])
        result = result.iloc[indexes]

        # Сортируем результаты в порядке отправки
        result = result.sort_values('date').reset_index(drop=True)

        # Выводим результаты
        if self.__verbose:
            print(f'{result.shape[0]} out of {len(descriptions)} submission results have been received.')

        # Возвращаем результат
        return result
