AC_INIT([fakealgo], [0.1.2])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign 1.13 dist-xz subdir-objects -Wall -Werror])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE

OPENCV_REQUIRED=3.1.0

AC_CHECK_PROG(HAVE_PKGCONFIG, pkg-config, [ ], [
  AC_MSG_ERROR([You need to have pkg-config installed!])
])

AC_PROG_CXX

PKG_CHECK_MODULES(opencv, [
  opencv >= $OPENCV_REQUIRED
], [
  AC_SUBST([opencv_CFLAGS])
  AC_SUBST([opencv_LIBS])
], [
  AC_MSG_ERROR([
    You need to install or upgrade opencv
  ])
])

AC_CONFIG_SRCDIR([src/fakealgo.cpp])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
