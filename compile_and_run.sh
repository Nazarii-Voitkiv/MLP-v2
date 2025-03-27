#!/bin/bash

echo "Компіляція класів..."

# Шлях до компілятора Java
JAVAC="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/javac"
JAVA="/Users/nazarii/Library/Java/JavaVirtualMachines/openjdk-23/Contents/Home/bin/java"

# Компілюємо всі Java файли в поточному каталозі
$JAVAC -d . *.java

if [ $? -eq 0 ]; then
    echo "Компіляція успішно завершена."
    echo "Запуск TestDataProcessor..."
    
    # Запуск програми з більшим розміром пам'яті (4 ГБ)
    $JAVA --enable-preview -Xmx4g -XX:+ShowCodeDetailsInExceptionMessages TestDataProcessor
else
    echo "Помилка компіляції."
fi
