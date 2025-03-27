#!/bin/bash

echo "Компіляція класів..."

# Шлях до компілятора Java
JAVAC="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/javac"
JAVA="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/java"

# Перевірка наявності Java
if [ ! -f "$JAVAC" ]; then
    echo "ПОМИЛКА: Java компілятор не знайдено за шляхом: $JAVAC"
    echo "Будь ласка, оновіть шлях до компілятора в скрипті або встановіть Java."
    exit 1
fi

if [ ! -f "$JAVA" ]; then
    echo "ПОМИЛКА: Java runtime не знайдено за шляхом: $JAVA"
    echo "Будь ласка, оновіть шлях до Java в скрипті або встановіть Java."
    exit 1
fi

# Компілюємо всі Java файли в поточному каталозі
$JAVAC -d . *.java

if [ $? -eq 0 ]; then
    echo "Компіляція успішно завершена."
    
    # Вибір класу для запуску (за замовчуванням - Main)
    CLASS_TO_RUN=${1:-Main}
    
    # Якщо аргумент "draw", запускаємо інтерфейс для малювання
    if [ "$1" == "draw" ]; then
        CLASS_TO_RUN="LetterDrawingApp"
    fi
    
    echo "Запуск $CLASS_TO_RUN..."
    
    # Запуск програми з більшим розміром пам'яті (8GB замість 4GB)
    $JAVA --enable-preview -Xmx8g -XX:+ShowCodeDetailsInExceptionMessages $CLASS_TO_RUN ${@:2}
else
    echo "Помилка компіляції."
fi
