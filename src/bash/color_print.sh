CR_RED="\033[1;31m";
CR_YELLOW="\033[1;33m";
CR_CYAN="\033[1;36m";
CR_GREEN="\033[1;92m";
CR_BLUE="\033[1;94m";
CR_MAGENTA="\033[1;95m";
CR_END="\033[00m";

function colorPrint {
  if [[ ${#} > 1 ]]
  then
    case $1 in
      w|W) color=${CR_YELLOW};;
      m|M) color=${CR_BLUE};;
      p|P) color=${CR_CYAN};;
      e|E) color=${CR_RED};;
      *) color=${CR_END};;
    esac
    old_color=${CR_END};
    echo "\n${color}$2${old_color}";
  fi
}

