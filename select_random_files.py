# Cleo Aguiar
# Script para compactar arquivos aleatórios contidos em um diretório
# Verificar se o script é o último arquivo do diretório, caso contrário
# será necessário alterar a linha random.shuffle(files_list[:-1]).
# variável number_files deve ser alterada conforme a necessidade.

import os
import random
import zipfile

number_files = 1500

files_list = os.listdir()[:-1]
random.shuffle(files_list)

with zipfile.ZipFile('random_files.zip', 'w') as new_zip:
   for name in files_list[:number_files]:
       new_zip.write(name)
