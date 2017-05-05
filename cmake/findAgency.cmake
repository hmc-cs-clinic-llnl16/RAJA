MRAI
# ---------
#
# Find Agency library
#
# This module finds an installed version of agency
#
# This module sets the following variables
#
# ::
#
#   Agency_FOUND - set to true if the SAMRAI library is found
#   Agency_INCLUDE_DIRS - directory containing the SAMRAI header files


include (FindPackageHandleStandardArgs)

find_path (AGENCY_PREFIX_INCLUDE_DIRS
  NAMES agency.hpp
)



find_package_handle_standard_args (SAMRAI
  DEFAULT_MSG
  AGENCY_INCLUDE_DIRS
)

mark_as_advanced (
  AGENCY_INCLUDE_DIRS
)

set (AGENCY_FOUND TRUE)
