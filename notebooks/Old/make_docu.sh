#!/bin/bash
languages=("es" "ja")

./make.bat html
./make.bat gettext

for lang in "${languages[@]}"; do
    sphinx-intl update -p _build/gettext -l $lang
    sphinx-build -b html -D language=$lang . ./_build/$lang
    # make 
done

# #!/bin/bash
# languages=("en" "fr" "es" "de")

# for lang in "${languages[@]}"; do
#     sphinx-build -b html -D language=$lang ./source ./build/$lang
# done
