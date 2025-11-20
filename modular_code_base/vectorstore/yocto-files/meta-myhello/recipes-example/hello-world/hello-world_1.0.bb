DESCRIPTION = "A simple 'Hello World' application"
SECTION = "examples"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

PV = "0.3"
# Specify the source files (our hello.c and Makefile from the 'files' subdirectory)
SRC_URI = "file://hello.c \
           file://Makefile"

# Where the source files will be extracted in the build directory
S = "${WORKDIR}"
CLEANBROKEN = "1"

# Define the compile step (using our Makefile)
do_compile() {
    oe_runmake
}

# Define the install step (installing the compiled binary to /usr/bin)
do_install() {
    install -d ${D}${bindir}
    install -m 0755 hello ${D}${bindir}
} 
