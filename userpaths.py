import os

alek_paths = {
    'train_folder': r'C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\train_data',
    'test_folder': r'C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\test_data',
    'validation_folder': r'C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning\clean_data\validation_data',
    'model_saves': r'C:\Users\aleks\OneDrive\Skole\DTU\7. Semester\Deep Learning'
    }

jonas_paths = {
    'train_folder': r'C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\data\clean_data\train_data',
    'test_folder': r'C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\data\clean_data\test_data',
    'validation_folder': r'C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\data\clean_data\validation_data',
    'model_saves': r'C:\Users\jonas\OneDrive\Desktop\DeepLearningProject\models'
    }

marcus_paths = {
    'train_folder': r'C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\train_data',
    'test_folder': r'C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\test_data',
    'validation_folder': r'C:\Users\Marcu\OneDrive - Danmarks Tekniske Universitet\DTU\Kandidat\1. Semester\Deep Learning\clean_data\validation_data',
    'model_saves': r''
    }

userpaths = {
    'jonas': jonas_paths,
    'Marcus': marcus_paths,
    'Marcu' : marcus_paths,
    'aleks': alek_paths
    }

user = os.getlogin()

train_folder = userpaths[user]['train_folder']
test_folder = userpaths[user]['test_folder']
validation_folder = userpaths[user]['validation_folder']
model_saves = userpaths[user]['model_saves']