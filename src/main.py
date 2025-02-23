import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class TitanicDataAnalyzer:
    """
    Класс для анализа данных о пассажирах Титаника,
    включая загрузку, предварительный анализ и разбиение на обучающую и тестовую выборки.
    """

    def __init__(self, url):
        """
        Инициализирует анализатор данных Titanic.

        Args:
            url (str): URL-адрес CSV-файла с данными.
        """
        self.url = url
        self.df = None  # DataFrame для хранения данных
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data(self):
        """
        Загружает данные из CSV-файла по указанному URL.
        """
        try:
            self.df = pd.read_csv(self.url)
            print("Данные успешно загружены.")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            self.df = None


    def perform_preliminary_analysis(self):
        """
        Выполняет предварительный анализ данных,
        включая определение количества признаков, типов данных и наличия пропущенных значений.
        """
        if self.df is None:
            print("Данные не загружены. Сначала загрузите данные.")
            return

        print("\n--- Предварительный анализ данных ---")
        print(f"Количество признаков: {self.df.shape[1]}")
        print("\nТипы данных:")
        print(self.df.dtypes)
        print("\nПропущенные значения:")
        print(self.df.isnull().sum())


    def split_data(self, test_size=0.2, random_state=42):
        """
        Разбивает данные на обучающую и тестовую выборки.

        Args:
            test_size (float): Доля тестовой выборки (по умолчанию 0.2).
            random_state (int): Зерно для случайного разбиения (по умолчанию 42).
        """
        if self.df is None:
            print("Данные не загружены. Сначала загрузите данные.")
            return

        try:
            X = self.df.drop('Survived', axis=1)  # Признаки
            y = self.df['Survived']  # Целевая переменная

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            print("\n--- Разбиение данных ---")
            print(f"Размер обучающей выборки: {len(self.X_train)}")
            print(f"Размер тестовой выборки: {len(self.X_test)}")
        except Exception as e:
            print(f"Ошибка при разбиении данных: {e}")
            self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None


    def visualize_data_split(self):
        """
        Визуализирует распределение целевой переменной в обучающей и тестовой выборках.
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            print("Данные не разбиты. Сначала разбейте данные.")
            return

        plt.figure(figsize=(12, 6), num='Распределение выживших в выборках')

        plt.subplot(1, 2, 1)
        sns.countplot(x=self.y_train)
        plt.title('Распределение выживших в обучающей выборке', fontsize=14, color='navy')
        plt.xlabel('Выжил (0 - Нет, 1 - Да)', fontsize=12)
        plt.ylabel('Количество', fontsize=12)

        plt.subplot(1, 2, 2)
        sns.countplot(x=self.y_test)
        plt.title('Распределение выживших в тестовой выборке', fontsize=14, color='navy')
        plt.xlabel('Выжил (0 - Нет, 1 - Да)', fontsize=12)
        plt.ylabel('Количество', fontsize=12)

        plt.suptitle('Визуализация разбиения данных', fontsize=16, color='darkgreen')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Для предотвращения перекрытия заголовков
        plt.show()


if __name__ == '__main__':
    titanic_url = 'https://github.com/datasciencedojo/datasets/blob/master/titanic.csv?raw=true'
    
    analyzer = TitanicDataAnalyzer(titanic_url)
    analyzer.load_data()
    analyzer.perform_preliminary_analysis()
    analyzer.split_data()
    analyzer.visualize_data_split()