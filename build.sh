set -eu

if [ ! -z ${1+x} ] && [ $1 == "clean" ]; then
	rm -rf build/
	rm *.bin
	exit 0
fi

# git submodule update --init --recursive

mkdir -p build

CXX=g++
CPPFLAGS="--std=c++14 -stdlib=libc++ -Wall -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -g -DNDEBUG"
CPPFLAGS="$CPPFLAGS -O2"
COMPILE_FLAGS="$CPPFLAGS -I libs -I libs/emilib -I /opt/local/include/eigen3"
LDLIBS="-lstdc++ -lpthread -ldl"
LDLIBS="$LDLIBS -lsdl2 -lglew -framework OpenGL"
# LDLIBS="$LDLIBS -ljemalloc"
OBJECTS=""

for source_path in src/*.cpp; do
	rel_source_path=${source_path#src/} # Remove src/ path prefix
	obj_path="build/${rel_source_path%.cpp}.o"
	OBJECTS="$OBJECTS $obj_path"
	if [ ! -f $obj_path ] || [ $obj_path -ot $source_path ]; then
		echo >&2 "Compiling $source_path to $obj_path..."
		$CXX $COMPILE_FLAGS -c $source_path -o $obj_path &
	fi
done

wait

echo >&2 "Linking..."
$CXX $CPPFLAGS $OBJECTS $LDLIBS -o sdf.bin
