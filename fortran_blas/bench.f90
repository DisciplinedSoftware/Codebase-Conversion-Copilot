program blas_bench
  implicit none

  integer, parameter :: nwarm = 3
  integer, parameter :: niter = 13
  integer, parameter :: nsizes = 3
  integer :: sizes(nsizes)
  data sizes / 64, 512, 2048 /

  ! External BLAS function declarations
  double precision, external :: dasum, ddot, dnrm2
  real, external :: sasum, sdot, snrm2

  integer :: si, n, iter
  double precision :: t0, t1, dt, t_total, t_sq, mean_ns, std_ns
  real :: st0, st1, sdt, st_total, st_sq, smean_ns, sstd_ns

  ! Working arrays (max size 2048*2048)
  integer, parameter :: maxn = 2048
  double precision :: DA(maxn*maxn), DB(maxn*maxn), DC(maxn*maxn)
  double precision :: DX(maxn), DY(maxn)
  real :: SA(maxn*maxn), SB(maxn*maxn), SC(maxn*maxn)
  real :: SX(maxn), SY(maxn)
  double precision :: dalpha, dbeta
  real :: salpha, sbeta
  double precision :: dres
  real :: sres

  integer :: i, j
  logical :: first_entry
  first_entry = .true.

  dalpha = 1.5d0
  dbeta  = 0.5d0
  salpha = 1.5e0
  sbeta  = 0.5e0

  write(*,'(A)') '['

  do si = 1, nsizes
    n = sizes(si)

    ! Initialize arrays
    do j = 1, n
      do i = 1, n
        DA(i + (j-1)*n) = sin(dble(i)*dble(j)*0.1d0)
        DB(i + (j-1)*n) = sin(dble(i+1)*dble(j)*0.1d0)
        DC(i + (j-1)*n) = sin(dble(i)*dble(j+1)*0.1d0)
        SA(i + (j-1)*n) = sin(real(i)*real(j)*0.1e0)
        SB(i + (j-1)*n) = sin(real(i+1)*real(j)*0.1e0)
        SC(i + (j-1)*n) = sin(real(i)*real(j+1)*0.1e0)
      end do
      DX(j) = dble(j) / dble(n)
      DY(j) = dble(n+1-j) / dble(n)
      SX(j) = real(j) / real(n)
      SY(j) = real(n+1-j) / real(n)
    end do

    ! Fix diagonal for triangular routines: large diagonal
    do i = 1, n
      DA(i + (i-1)*n) = dble(n) + dble(i)
      SA(i + (i-1)*n) = real(n) + real(i)
    end do

    !---------------------------------------------------------------------------
    ! Level 1: dasum
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      dres = dasum(n, DX, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    if (.not. first_entry) write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dasum", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'
    first_entry = .false.

    !---------------------------------------------------------------------------
    ! Level 1: sasum
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      sres = sasum(n, SX, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sasum", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 1: daxpy
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call daxpy(n, dalpha, DX, 1, DY, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "daxpy", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 1: saxpy
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call saxpy(n, salpha, SX, 1, SY, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "saxpy", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 1: ddot
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      dres = ddot(n, DX, 1, DY, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "ddot", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 1: sdot
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      sres = sdot(n, SX, 1, SY, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sdot", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 1: dnrm2
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      dres = dnrm2(n, DX, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dnrm2", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 1: snrm2
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      sres = snrm2(n, SX, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "snrm2", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 1: dscal
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dscal(n, dalpha, DX, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dscal", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 1: sscal
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call sscal(n, salpha, SX, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sscal", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 1: dswap
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dswap(n, DX, 1, DY, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dswap", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 1: sswap
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call sswap(n, SX, 1, SY, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sswap", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 2: dgemv
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dgemv('N', n, n, dalpha, DA, n, DX, 1, dbeta, DY, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dgemv", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 2: sgemv
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call sgemv('N', n, n, salpha, SA, n, SX, 1, sbeta, SY, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sgemv", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 2: dger
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dger(n, n, dalpha, DX, 1, DY, 1, DA, n)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dger", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 2: sger
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call sger(n, n, salpha, SX, 1, SY, 1, SA, n)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sger", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 2: dsymv
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dsymv('U', n, dalpha, DA, n, DX, 1, dbeta, DY, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dsymv", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 2: ssymv
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call ssymv('U', n, salpha, SA, n, SX, 1, sbeta, SY, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "ssymv", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 2: dtrmv
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dtrmv('U', 'N', 'N', n, DA, n, DX, 1)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dtrmv", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 2: strmv
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call strmv('U', 'N', 'N', n, SA, n, SX, 1)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "strmv", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 3: dgemm
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dgemm('N', 'N', n, n, n, dalpha, DA, n, DB, n, dbeta, DC, n)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dgemm", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 3: sgemm
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call sgemm('N', 'N', n, n, n, salpha, SA, n, SB, n, sbeta, SC, n)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "sgemm", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 3: dsyrk
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dsyrk('U', 'N', n, n, dalpha, DA, n, dbeta, DC, n)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dsyrk", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 3: ssyrk
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call ssyrk('U', 'N', n, n, salpha, SA, n, sbeta, SC, n)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "ssyrk", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

    !---------------------------------------------------------------------------
    ! Level 3: dtrmm
    !---------------------------------------------------------------------------
    t_total = 0.0d0; t_sq = 0.0d0
    do iter = 1, niter
      call cpu_time(t0)
      call dtrmm('L', 'U', 'N', 'N', n, n, dalpha, DA, n, DB, n)
      call cpu_time(t1)
      dt = (t1 - t0) * 1.0d9
      if (iter > nwarm) then
        t_total = t_total + dt
        t_sq    = t_sq + dt*dt
      end if
    end do
    mean_ns = t_total / dble(niter - nwarm)
    std_ns  = sqrt(max(0.0d0, t_sq/dble(niter-nwarm) - mean_ns*mean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "dtrmm", "precision": "f64", "n": ', n, &
      ', "mean_ns": ', mean_ns, ', "std_ns": ', std_ns, '}'

    !---------------------------------------------------------------------------
    ! Level 3: strmm
    !---------------------------------------------------------------------------
    st_total = 0.0e0; st_sq = 0.0e0
    do iter = 1, niter
      call cpu_time(st0)
      call strmm('L', 'U', 'N', 'N', n, n, salpha, SA, n, SB, n)
      call cpu_time(st1)
      sdt = (st1 - st0) * 1.0e9
      if (iter > nwarm) then
        st_total = st_total + sdt
        st_sq    = st_sq + sdt*sdt
      end if
    end do
    smean_ns = st_total / real(niter - nwarm)
    sstd_ns  = sqrt(max(0.0e0, st_sq/real(niter-nwarm) - smean_ns*smean_ns))
    write(*,'(A)') ','
    write(*,'(A,I0,A,F20.3,A,F20.3,A)') &
      '  {"routine": "strmm", "precision": "f32", "n": ', n, &
      ', "mean_ns": ', dble(smean_ns), ', "std_ns": ', dble(sstd_ns), '}'

  end do ! sizes

  write(*,'(A)') ']'

contains

  function itoa(val) result(str)
    integer, intent(in) :: val
    character(len=20) :: str
    write(str, '(I0)') val
    str = trim(adjustl(str))
  end function itoa

end program blas_bench
