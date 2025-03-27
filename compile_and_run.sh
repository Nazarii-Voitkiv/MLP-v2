#!/bin/bash

echo "Компіляція класів..."

# Шлях до компілятора Java
JAVAC="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/javac"
JAVA="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/java"

# Компілюємо всі Java файли в поточному каталозі
$JAVAC -d . *.java

if [ $? -eq 0 ]; then
    echo "Компіляція успішно завершена."
    
    # Вибір класу для запуску (за замовчуванням - LetterClassificationTest)
    CLASS_TO_RUN=${1:-LetterClassificationTest}
    echo "Запуск $CLASS_TO_RUN..."
    
    # Запуск програми з більшим розміром пам'яті
    $JAVA --enable-preview -Xmx4g -XX:+ShowCodeDetailsInExceptionMessages $CLASS_TO_RUN
else
    echo "Помилка компіляції."
fi
