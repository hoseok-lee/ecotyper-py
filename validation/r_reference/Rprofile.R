# Used only for generating R reference outputs (R_PROFILE_USER).
# EcoTyper's recovery scripts call NMF::rnmf() without seeding R's RNG, so their
# output depends on a time/PID-derived seed. Seeding once at R start-up makes the
# reference reproducible; the Python port takes the same integer as `seed`.
local({
  s <- Sys.getenv("ECOTYPER_REF_SEED")
  if (nzchar(s)) set.seed(as.integer(s))
})
