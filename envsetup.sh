current_dir=$(pwd)
echo "Current directory: $current_dir"

venv_dir="$current_dir/venv"
if [ ! -d "$venv_dir" ]; then
	echo "Creating virtual environment in $venv_dir"
	python3 -m venv "$venv_dir"
	"$venv_dir/bin/pip" install --upgrade pip
	"$venv_dir/bin/pip" install -r "$current_dir/requirements.txt"
else
	echo "Virtual environment already exists in $venv_dir"
fi

source "$venv_dir/bin/activate"
echo "Virtual environment activated."
